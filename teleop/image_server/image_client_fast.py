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
    def __init__(self, tv_img_shape = None, tv_img_shm_name = None, wrist_img_shape = None, wrist_img_shm_name = None, 
                       image_show = False, server_address = "192.168.1.224", port = 5556, Unit_Test = False,
                       use_threading = True, drop_old_frames = True, decode_quality = cv2.IMREAD_COLOR):
        """
        tv_img_shape: User's expected head camera resolution shape (H, W, C). It should match the output of the image service terminal.

        tv_img_shm_name: Shared memory is used to easily transfer images across processes to the Vuer.

        wrist_img_shape: User's expected wrist camera resolution shape (H, W, C). It should maintain the same shape as tv_img_shape.

        wrist_img_shm_name: Shared memory is used to easily transfer images.
        
        image_show: Whether to display received images in real time.

        server_address: The ip address to execute the image server script.

        port: The port number to bind to. It should be the same as the image server.

        Unit_Test: When both server and client are True, it can be used to test the image transfer latency, \
                   network jitter, frame loss rate and other information.
                   
        use_threading: Use separate thread for processing to prevent blocking.
        
        drop_old_frames: Always process the latest frame, dropping old ones.
        
        decode_quality: cv2.IMREAD_COLOR or cv2.IMREAD_REDUCED_COLOR_2 for faster decoding.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
        self._use_threading = use_threading
        self._drop_old_frames = drop_old_frames
        self._decode_quality = decode_quality

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = self.tv_image_shm.buf)
            self.tv_enable_shm = True
        
        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = self.wrist_image_shm.buf)
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

            print(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, \
                  Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%, Dropped: {self._dropped_frames}")
    
    def _close(self):
        if self._use_threading:
            self._process_thread_running = False
        self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        print("Image client has been closed.")

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
        # Set up ZeroMQ context and socket with optimizations
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        
        # Socket optimizations for low latency
        self._socket.setsockopt(zmq.RCVHWM, 1)  # Receive high water mark - drop old messages
        self._socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self._socket.setsockopt(zmq.RCVTIMEO, 100)  # Receive timeout in ms
        
        # TCP optimizations
        self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 120)
        
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Start processing thread if enabled
        if self._use_threading:
            process_thread = threading.Thread(target=self._process_frames)
            process_thread.daemon = True
            process_thread.start()

        print("\nImage client has started, waiting to receive data...")
        try:
            while self.running:
                try:
                    # Receive message with timeout
                    message = self._socket.recv(zmq.NOBLOCK if self._drop_old_frames else 0)
                except zmq.Again:
                    # No message available
                    time.sleep(0.001)
                    continue
                except zmq.error.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        continue
                    else:
                        raise
                
                receive_time = time.time()

                # Parse header if in test mode
                if self._enable_performance_eval:
                    header_size = struct.calcsize('dI')
                    try:
                        header = message[:header_size]
                        jpg_bytes = message[header_size:]
                        timestamp, frame_id = struct.unpack('dI', header)
                    except struct.error as e:
                        print(f"[Image Client] Error unpacking header: {e}, discarding message.")
                        continue
                else:
                    jpg_bytes = message
                    timestamp = frame_id = None
                
                # Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, self._decode_quality)
                if current_image is None:
                    print("[Image Client] Failed to decode image.")
                    continue

                if self._use_threading:
                    # Queue frame for processing
                    frame_data = (current_image, receive_time, timestamp, frame_id)
                    
                    if self._drop_old_frames and self._frame_queue.full():
                        # Drop the old frame
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
                    # Process directly (original behavior)
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

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()

if __name__ == "__main__":
    # Example with optimizations enabled
    client = ImageClient(
        image_show=True, 
        server_address='192.168.1.224', 
        Unit_Test=False,
        use_threading=True,  # Enable threading for non-blocking processing
        drop_old_frames=True,  # Always use latest frame
        decode_quality=cv2.IMREAD_COLOR  # Use cv2.IMREAD_REDUCED_COLOR_2 for faster decoding
    )
    client.receive_process()