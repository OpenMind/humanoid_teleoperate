import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import threading
import queue


class ImageClient:
    def __init__(self, tv_img_shape=None, tv_img_shm_name=None, wrist_img_shape=None, wrist_img_shm_name=None, 
                 image_show=False, server_address="192.168.1.224", port=5556, Unit_Test=False,
                 use_threading=True, drop_old_frames=True, decode_quality=cv2.IMREAD_COLOR,
                 auto_reconnect=True, reconnect_interval=1.0, max_reconnect_interval=30.0):
        """
        tv_img_shape: User's expected head camera resolution shape (H, W, C). It should match the output of the image service terminal.
        tv_img_shm_name: Shared memory is used to easily transfer images across processes to the Vuer.
        wrist_img_shape: User's expected wrist camera resolution shape (H, W, C). It should maintain the same shape as tv_img_shape.
        wrist_img_shm_name: Shared memory is used to easily transfer images.
        image_show: Whether to display received images in real time.
        server_address: The ip address to execute the image server script.
        port: The port number to bind to. It should be the same as the image server.
        Unit_Test: When both server and client are True, it can be used to test the image transfer latency,
                   network jitter, frame loss rate and other information.
        use_threading: Use separate thread for processing to prevent blocking.
        drop_old_frames: Always process the latest frame, dropping old ones.
        decode_quality: cv2.IMREAD_COLOR or cv2.IMREAD_REDUCED_COLOR_2 for faster decoding.
        auto_reconnect: Automatically reconnect on connection loss.
        reconnect_interval: Initial reconnection interval in seconds.
        max_reconnect_interval: Maximum reconnection interval in seconds.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
        self._use_threading = use_threading
        self._drop_old_frames = drop_old_frames
        self._decode_quality = decode_quality
        self._auto_reconnect = auto_reconnect
        self._reconnect_interval = reconnect_interval
        self._max_reconnect_interval = max_reconnect_interval
        
        # Connection state
        self._connected = False
        self._socket = None
        self._context = None

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=self.tv_image_shm.buf)
            self.tv_enable_shm = True
        
        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        # Threading setup
        if self._use_threading:
            self._frame_queue = queue.Queue(maxsize=1 if self._drop_old_frames else 10)
            self._process_thread_running = True

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Data transmission quality metrics
        self._latencies = deque()  # Latencies of frames within the time window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs
        self._dropped_frames = 0  # Frames dropped due to queue overflow

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency
        latency = receive_time - timestamp
        self._latencies.append(latency)

        # Remove latencies outside the time window
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # Update frame counts for lost frame calculation
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # Calculate latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, "
                  f"Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%, Dropped: {self._dropped_frames}")

    def _create_socket(self):
        """Create and configure ZMQ socket"""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
                
        if not self._context:
            self._context = zmq.Context()
            
        self._socket = self._context.socket(zmq.SUB)
        
        # Socket optimizations for low latency
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        
        # TCP optimizations
        self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 10)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)
        
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
    def _connect(self):
        """Establish connection to server"""
        try:
            self._create_socket()
            self._socket.connect(f"tcp://{self._server_address}:{self._port}")
            self._connected = True
            print(f"[Image Client] Connected to {self._server_address}:{self._port}")
            return True
        except Exception as e:
            print(f"[Image Client] Connection failed: {e}")
            self._connected = False
            return False
            
    def _reconnect(self):
        """Handle reconnection with exponential backoff"""
        reconnect_delay = self._reconnect_interval
        
        while self.running and not self._connected:
            print(f"[Image Client] Attempting to reconnect to {self._server_address}:{self._port}...")
            
            if self._connect():
                print("[Image Client] Reconnection successful!")
                return True
                
            print(f"[Image Client] Reconnection failed, waiting {reconnect_delay:.1f} seconds...")
            
            # Wait with ability to interrupt
            wait_time = 0
            while wait_time < reconnect_delay and self.running:
                time.sleep(0.1)
                wait_time += 0.1
                
            # Exponential backoff
            reconnect_delay = min(reconnect_delay * 2, self._max_reconnect_interval)
            
        return False
    
    def _close(self):
        """Clean shutdown"""
        self.running = False
        self._connected = False
        
        if self._use_threading:
            self._process_thread_running = False
            
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
                
        if self._context:
            try:
                self._context.term()
            except:
                pass
                
        if self._image_show:
            cv2.destroyAllWindows()
            
        print("[Image Client] Closed")

    def _process_frames(self):
        """Process frames in a separate thread"""
        while self._process_thread_running:
            try:
                frame_data = self._frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue
                
                current_image, receive_time, timestamp, frame_id = frame_data
                
                # Copy to shared memory
                if self.tv_enable_shm:
                    np.copyto(self.tv_img_array, np.array(current_image[:, :self.tv_img_shape[1]]))
                
                if self.wrist_enable_shm:
                    np.copyto(self.wrist_img_array, np.array(current_image[:, -self.wrist_img_shape[1]:]))
                
                # Display if enabled
                if self._image_show:
                    height, width = current_image.shape[:2]
                    resized_image = cv2.resize(current_image, (width // 2, height // 2))
                    cv2.imshow('Image Client Stream', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                
                # Update metrics if enabled
                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Image Client] Error in processing thread: {e}")
    
    def receive_process(self):
        """Main receive process with automatic reconnection"""
        # Initial connection
        if not self._connect():
            if self._auto_reconnect:
                if not self._reconnect():
                    print("[Image Client] Failed to establish initial connection")
                    return
            else:
                print("[Image Client] Initial connection failed, auto-reconnect disabled")
                return
                
        # Start processing thread if enabled
        if self._use_threading:
            process_thread = threading.Thread(target=self._process_frames)
            process_thread.daemon = True
            process_thread.start()
            
        print("\n[Image Client] Started, receiving data...")
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while self.running:
                try:
                    # Check if we're connected
                    if not self._connected:
                        if self._auto_reconnect:
                            if not self._reconnect():
                                break
                        else:
                            print("[Image Client] Connection lost, auto-reconnect disabled")
                            break
                            
                    # Try to receive message
                    try:
                        message = self._socket.recv(zmq.NOBLOCK if self._drop_old_frames else 0)
                        consecutive_errors = 0  # Reset error counter on success
                        
                    except zmq.Again:
                        # No message available
                        time.sleep(0.001)
                        continue
                        
                    except zmq.error.ZMQError as e:
                        if e.errno == zmq.EAGAIN:
                            continue
                        elif e.errno in [zmq.ECONNREFUSED, zmq.EHOSTUNREACH, zmq.ETIMEDOUT]:
                            # Connection issues
                            print(f"[Image Client] Connection error: {e}")
                            self._connected = False
                            continue
                        else:
                            raise
                            
                    receive_time = time.time()
                    
                    # Parse and process message
                    if self._enable_performance_eval:
                        header_size = struct.calcsize('dI')
                        try:
                            header = message[:header_size]
                            jpg_bytes = message[header_size:]
                            timestamp, frame_id = struct.unpack('dI', header)
                        except struct.error as e:
                            print(f"[Image Client] Error unpacking header: {e}")
                            continue
                    else:
                        jpg_bytes = message
                        timestamp = frame_id = None
                        
                    # Decode image
                    np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    current_image = cv2.imdecode(np_img, self._decode_quality)
                    
                    if current_image is None:
                        print("[Image Client] Failed to decode image")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print("[Image Client] Too many decode errors, reconnecting...")
                            self._connected = False
                        continue
                        
                    # Process frame
                    if self._use_threading:
                        frame_data = (current_image, receive_time, timestamp, frame_id)
                        
                        if self._drop_old_frames and self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                                if self._enable_performance_eval:
                                    self._dropped_frames += 1
                            except queue.Empty:
                                pass
                                
                        try:
                            self._frame_queue.put_nowait(frame_data)
                        except queue.Full:
                            if self._enable_performance_eval:
                                self._dropped_frames += 1
                    else:
                        # Direct processing
                        if self.tv_enable_shm:
                            np.copyto(self.tv_img_array, np.array(current_image[:, :self.tv_img_shape[1]]))
                            
                        if self.wrist_enable_shm:
                            np.copyto(self.wrist_img_array, np.array(current_image[:, -self.wrist_img_shape[1]:]))
                            
                        if self._image_show:
                            height, width = current_image.shape[:2]
                            resized_image = cv2.resize(current_image, (width // 2, height // 2))
                            cv2.imshow('Image Client Stream', resized_image)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.running = False
                                
                        if self._enable_performance_eval:
                            self._update_performance_metrics(timestamp, frame_id, receive_time)
                            self._print_performance_metrics(receive_time)
                            
                except zmq.ZMQError as e:
                    print(f"[Image Client] ZMQ error: {e}")
                    self._connected = False
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("[Image Client] Too many consecutive errors")
                        if not self._auto_reconnect:
                            break
                            
                except Exception as e:
                    print(f"[Image Client] Unexpected error: {e}")
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("[Image Client] Too many consecutive errors")
                        if not self._auto_reconnect:
                            break
                            
        except KeyboardInterrupt:
            print("[Image Client] Interrupted by user")
        finally:
            self._close()


if __name__ == "__main__":
    # Example with optimizations and auto-reconnection enabled
    client = ImageClient(
        image_show=True,
        server_address='10.0.0.73',
        Unit_Test=False,
        use_threading=True,
        drop_old_frames=True,
        decode_quality=cv2.IMREAD_COLOR,
        auto_reconnect=True,  # Enable auto-reconnection
        reconnect_interval=1.0,  # Start with 1 second retry
        max_reconnect_interval=30.0  # Max 30 seconds between retries
    )
    client.receive_process()