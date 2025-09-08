import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import platform
import threading
import queue
import subprocess
import re


class OpenCVCamera():
    def __init__(self, device_id, img_shape, fps):
        """
        device_id: /dev/video* or *
        img_shape: [height, width]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.device_name = "Unknown"
        
        # Use appropriate backend based on OS
        system = platform.system()
        if system == "Darwin":  # macOS
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_AVFOUNDATION)
        elif system == "Linux":
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        else:  # Windows or others
            self.cap = cv2.VideoCapture(self.id)
            
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Reduce buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get device name
        self._get_device_name(system)
        
        # Print camera information
        self._print_camera_info()

        # Test if the camera can read frames
        if not self._can_read_frame():
            print(f"[Image Server] Camera {self.id} Error: Failed to initialize the camera or read frames. Exiting...")
            self.release()

    def _get_device_name(self, system):
        """Get the device name based on the operating system"""
        try:
            if system == "Linux":
                # Try to get device name using v4l2-ctl
                try:
                    if isinstance(self.id, str) and '/dev/video' in self.id:
                        device_path = self.id
                    else:
                        device_path = f'/dev/video{self.id}'
                    
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', device_path, '--info'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    
                    if result.returncode == 0:
                        # Parse the output for card type
                        for line in result.stdout.split('\n'):
                            if 'Card type' in line:
                                self.device_name = line.split(':', 1)[1].strip()
                                break
                except (subprocess.SubprocessError, FileNotFoundError):
                    # v4l2-ctl not available, try alternative method
                    self._get_device_name_from_sys(system)
                    
            elif system == "Darwin":  # macOS
                # For macOS, we can try to get device name differently
                # This is limited by OpenCV's API
                backend_name = self.cap.getBackendName()
                self.device_name = f"Camera {self.id} ({backend_name})"
                
            elif system == "Windows":
                # For Windows, try to get more info
                backend_name = self.cap.getBackendName()
                self.device_name = f"Camera {self.id} ({backend_name})"
                
        except Exception as e:
            print(f"[Image Server] Could not get device name: {e}")
            self.device_name = f"Camera {self.id}"

    def _get_device_name_from_sys(self, system):
        """Alternative method to get device name from /sys filesystem (Linux only)"""
        if system == "Linux":
            try:
                if isinstance(self.id, str) and '/dev/video' in self.id:
                    video_num = self.id.replace('/dev/video', '')
                else:
                    video_num = str(self.id)
                    
                # Try to read from /sys/class/video4linux/
                sys_path = f'/sys/class/video4linux/video{video_num}/name'
                with open(sys_path, 'r') as f:
                    self.device_name = f.read().strip()
            except:
                self.device_name = f"Camera {self.id}"

    def _print_camera_info(self):
        """Print detailed camera information"""
        print("\n" + "="*60)
        print(f"[Camera Information - Device {self.id}]")
        print("="*60)
        
        # Device name
        print(f"Device Name: {self.device_name}")
        print(f"Device ID: {self.id}")
        
        # Requested vs Actual resolution
        requested_width = self.img_shape[1]
        requested_height = self.img_shape[0]
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nResolution:")
        print(f"  Requested: {requested_width}x{requested_height}")
        print(f"  Actual:    {actual_width}x{actual_height}")
        
        if (actual_width != requested_width) or (actual_height != requested_height):
            print(f"  ⚠️  Warning: Actual resolution differs from requested!")
        
        # FPS
        requested_fps = self.fps
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"\nFrame Rate:")
        print(f"  Requested: {requested_fps} FPS")
        print(f"  Actual:    {actual_fps:.2f} FPS")
        
        # Other camera properties
        print(f"\nAdditional Properties:")
        
        # Backend
        backend = self.cap.getBackendName()
        print(f"  Backend: {backend}")
        
        # FOURCC codec
        fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
        print(f"  FOURCC: {fourcc}")
        
        # Buffer size
        buffer_size = int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE))
        print(f"  Buffer Size: {buffer_size}")
        
        # Exposure and brightness if available
        try:
            exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            if exposure != -1:
                print(f"  Exposure: {exposure}")
        except:
            pass
            
        try:
            brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            if brightness != -1:
                print(f"  Brightness: {brightness}")
        except:
            pass
        
        # Auto exposure
        try:
            auto_exposure = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            print(f"  Auto Exposure: {auto_exposure}")
        except:
            pass
        
        print("="*60 + "\n")

    def get_actual_resolution(self):
        """Return the actual resolution as tuple (width, height)"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def release(self):
        self.cap.release()

    def get_frame(self):
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image


class ImageServer:
    def __init__(self, config, port = 5556, Unit_Test = False):
        print("\n" + "#"*60)
        print("#" + " "*20 + "IMAGE SERVER" + " "*26 + "#")
        print("#"*60)
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()
        
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])
        
        # Performance optimization settings
        self.jpeg_quality = config.get('jpeg_quality', 50)
        self.skip_frames = config.get('skip_frames', 0)
        self.downscale_factor = config.get('downscale_factor', 1.0)
        self.use_threading = config.get('use_threading', True)
        self.buffer_size = config.get('buffer_size', 1)
        
        self.port = port
        self.Unit_Test = Unit_Test
        self.frame_counter = 0

        # Initialize head cameras
        self.head_cameras = []
        if self.head_camera_type == 'opencv':
            for i, device_id in enumerate(self.head_camera_id_numbers):
                print(f"\n[Initializing Camera {i+1}/{len(self.head_camera_id_numbers)}]")
                camera = OpenCVCamera(device_id=device_id, img_shape=self.head_image_shape, fps=self.fps)
                self.head_cameras.append(camera)
        else:
            print(f"[Image Server] Unsupported head_camera_type: {self.head_camera_type}")

        # Print summary of all cameras
        self._print_camera_summary()

        # Set ZeroMQ context and socket with optimizations
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # ZMQ optimizations
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        
        self.socket.bind(f"tcp://*:{self.port}")

        # Threading setup
        if self.use_threading:
            self.frame_queue = queue.Queue(maxsize=self.buffer_size)
            self.capture_thread_running = True

        if self.Unit_Test:
            self._init_performance_metrics()

        print("\n" + "+"*60)
        print(f"[Image Server] Server started successfully!")
        print(f"[Image Server] Listening on port: {self.port}")
        print(f"[Image Server] Waiting for client connections...")
        print("+"*60 + "\n")

    def _print_camera_summary(self):
        """Print a summary of all connected cameras"""
        print("\n" + "-"*60)
        print("CAMERA SUMMARY")
        print("-"*60)
        print(f"Total cameras connected: {len(self.head_cameras)}")
        
        for i, cam in enumerate(self.head_cameras):
            actual_res = cam.get_actual_resolution()
            print(f"\nCamera {i+1}:")
            print(f"  Name: {cam.device_name}")
            print(f"  ID: {cam.id}")
            print(f"  Resolution: {actual_res[0]}x{actual_res[1]}")
            print(f"  FPS: {cam.cap.get(cv2.CAP_PROP_FPS):.2f}")
        
        # Calculate and print combined resolution after concatenation
        if self.head_cameras:
            total_width = sum(cam.get_actual_resolution()[0] for cam in self.head_cameras)
            total_height = self.head_cameras[0].get_actual_resolution()[1]  # Assuming all same height
            print(f"\nCombined resolution (after concatenation): {total_width}x{total_height}")
            
            if self.downscale_factor < 1.0:
                scaled_width = int(total_width * self.downscale_factor)
                scaled_height = int(total_height * self.downscale_factor)
                print(f"After downscaling ({self.downscale_factor}x): {scaled_width}x{scaled_height}")
        
        print("-"*60)

    def _init_performance_metrics(self):
        self.frame_count = 0
        self.time_window = 1.0
        self.frame_times = deque()
        self.start_time = time.time()

    def _update_performance_metrics(self, current_time):
        self.frame_times.append(current_time)
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        if self.use_threading:
            self.capture_thread_running = False
        for cam in self.head_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def _capture_frames(self):
        """Capture frames in a separate thread"""
        while self.capture_thread_running:
            head_frames = []
            for cam in self.head_cameras:
                if self.head_camera_type == 'opencv':
                    color_image = cam.get_frame()
                    if color_image is None:
                        continue
                head_frames.append(color_image)
            
            if len(head_frames) == len(self.head_cameras):
                # Drop old frames if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(head_frames)

    def send_process(self):
        try:
            # Start capture thread if enabled
            if self.use_threading:
                capture_thread = threading.Thread(target=self._capture_frames)
                capture_thread.daemon = True
                capture_thread.start()

            while True:
                # Frame skipping
                if self.skip_frames > 0:
                    self.frame_counter += 1
                    if self.frame_counter % (self.skip_frames + 1) != 0:
                        continue

                # Get frames
                if self.use_threading:
                    try:
                        head_frames = self.frame_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                else:
                    head_frames = []
                    for cam in self.head_cameras:
                        if self.head_camera_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                print("[Image Server] Head camera frame read is error.")
                                break
                        head_frames.append(color_image)
                    
                    if len(head_frames) != len(self.head_cameras):
                        break

                # Concatenate frames
                head_color = cv2.hconcat(head_frames)
                
                # Downscale if needed
                if self.downscale_factor < 1.0:
                    new_width = int(head_color.shape[1] * self.downscale_factor)
                    new_height = int(head_color.shape[0] * self.downscale_factor)
                    head_color = cv2.resize(head_color, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                full_color = head_color

                # Encode with specified quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                ret, buffer = cv2.imencode('.jpg', full_color, encode_param)
                if not ret:
                    print("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                if self.Unit_Test:
                    timestamp = time.time()
                    frame_id = self.frame_count
                    header = struct.pack('dI', timestamp, frame_id)
                    message = header + jpg_bytes
                else:
                    message = jpg_bytes

                # Send without blocking
                try:
                    self.socket.send(message, zmq.NOBLOCK)
                except zmq.Again:
                    # Socket would block, skip this frame
                    pass

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            self._close()


def list_available_cameras():
    """List all available cameras on the system"""
    print("\n" + "="*60)
    print("SCANNING FOR AVAILABLE CAMERAS")
    print("="*60)
    
    available_cameras = []
    system = platform.system()
    
    # Test up to 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                print(f"\nFound Camera at index {i}:")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps:.2f}")
                print(f"  Backend: {backend}")
                
                available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("\nNo cameras found!")
    else:
        print(f"\nTotal cameras found: {len(available_cameras)}")
        print(f"Available indices: {available_cameras}")
    
    print("="*60 + "\n")
    return available_cameras


if __name__ == "__main__":
    # First, list all available cameras
    available = list_available_cameras()
    
    # Configuration
    config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [1080, 1920],  # Original resolution
        'head_camera_id_numbers': [0] if available else [],  # Use first available camera
        
        # Performance optimizations
        'jpeg_quality': 30,  # Reduce JPEG quality (1-100, lower = faster)
        'skip_frames': 2,  # Skip every other frame (0 = no skip, 1 = skip 1, etc.)
        'downscale_factor': 0.5,  # Downscale to 30% (1.0 = no downscale)
        'use_threading': True,  # Use separate thread for capture
        'buffer_size': 1,  # Minimal buffer to reduce latency
    }

    if config['head_camera_id_numbers']:
        server = ImageServer(config, Unit_Test=False)
        server.send_process()
    else:
        print("No cameras available to start the server.")