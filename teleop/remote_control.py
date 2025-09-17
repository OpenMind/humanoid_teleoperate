import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.robot_hand_brainco import Brainco_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from sshkeyboard import listen_keyboard, stop_listening

# for remote teleop
import asyncio
import websockets
import json
import msgpack
import queue
import threading
from typing import Optional, Dict, Any
from televuer import TeleData, TeleStateData

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int,publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
start_signal = False
running = True
should_toggle_recording = False
is_recording = False
def on_press(key):
    global running, start_signal, should_toggle_recording
    if key == 'c':
        start_signal = True
        logger_mp.info("Program start signal received.")
    elif key == 'q' and start_signal == True:
        stop_listening()
        running = False
    elif key == 's' and start_signal == True:
        should_toggle_recording = True
    else:
        logger_mp.info(f"{key} was pressed, but no action is defined for this key.")
listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press, "until": None, "sequential": False,}, daemon=True)
listen_keyboard_thread.start()

class WebSocketReceiver:
    """High-performance WebSocket client for receiving TeleData"""
    
    def __init__(self, uri="ws://localhost:8765", use_msgpack=True, 
                 queue_size=1, measure_latency=False, logger=None):
        self.uri = uri
        self.use_msgpack = use_msgpack
        self.measure_latency = measure_latency
        
        # Single-item queue for latest data only
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread = None
        self.loop = None
        
        # Performance tracking
        self.stats_lock = threading.Lock()
        self.receive_count = 0
        self.total_bytes_received = 0
        self.last_stats_time = time.time()
        self.latency_samples = []
        self.dropped_frames = 0
        self.last_frame_id = -1
        
    def deserialize_teledata(self, data_dict: Dict[str, Any]) -> TeleData:
        """Deserialize received dictionary back to TeleData dataclass"""
        
        # Deserialize TeleStateData first if present
        tele_state = None
        if 'tele_state' in data_dict and data_dict['tele_state'] is not None:
            state_dict = data_dict['tele_state']
            
            # Convert thumbstick values back to numpy arrays
            left_thumbstick = state_dict.get('left_thumbstick_value')
            if left_thumbstick is not None:
                left_thumbstick = np.array(left_thumbstick, dtype=np.float32)
            else:
                left_thumbstick = np.zeros(2, dtype=np.float32)
                
            right_thumbstick = state_dict.get('right_thumbstick_value')
            if right_thumbstick is not None:
                right_thumbstick = np.array(right_thumbstick, dtype=np.float32)
            else:
                right_thumbstick = np.zeros(2, dtype=np.float32)
            
            tele_state = TeleStateData(
                left_pinch_state=state_dict.get('left_pinch_state', 0),
                right_pinch_state=state_dict.get('right_pinch_state', 0),
                left_squeeze_state=state_dict.get('left_squeeze_state', 0),
                right_squeeze_state=state_dict.get('right_squeeze_state', 0),
                left_squeeze_value=state_dict.get('left_squeeze_value', 0.0),
                right_squeeze_value=state_dict.get('right_squeeze_value', 0.0),
                left_trigger_state=state_dict.get('left_trigger_state', False),
                right_trigger_state=state_dict.get('right_trigger_state', False),
                left_squeeze_ctrl_state=state_dict.get('left_squeeze_ctrl_state', False),
                right_squeeze_ctrl_state=state_dict.get('right_squeeze_ctrl_state', False),
                left_squeeze_ctrl_value=state_dict.get('left_squeeze_ctrl_value', 0.0),
                right_squeeze_ctrl_value=state_dict.get('right_squeeze_ctrl_value', 0.0),
                left_thumbstick_state=state_dict.get('left_thumbstick_state', False),
                right_thumbstick_state=state_dict.get('right_thumbstick_state', False),
                left_thumbstick_value=left_thumbstick,
                right_thumbstick_value=right_thumbstick,
                left_aButton=state_dict.get('left_aButton', False),
                right_aButton=state_dict.get('right_aButton', False),
                left_bButton=state_dict.get('left_bButton', False),
                right_bButton=state_dict.get('right_bButton', False)
            )
        else:
            tele_state = TeleStateData()
        
        # Convert pose matrices back to numpy arrays
        head_pose = np.array(data_dict['head_pose'], dtype=np.float32) if data_dict.get('head_pose') is not None else np.eye(4, dtype=np.float32)
        left_arm_pose = np.array(data_dict['left_arm_pose'], dtype=np.float32) if data_dict.get('left_arm_pose') is not None else np.eye(4, dtype=np.float32)
        right_arm_pose = np.array(data_dict['right_arm_pose'], dtype=np.float32) if data_dict.get('right_arm_pose') is not None else np.eye(4, dtype=np.float32)
        
        # Convert hand positions (optional fields)
        left_hand_pos = np.array(data_dict['left_hand_pos'], dtype=np.float32) if data_dict.get('left_hand_pos') is not None else None
        right_hand_pos = np.array(data_dict['right_hand_pos'], dtype=np.float32) if data_dict.get('right_hand_pos') is not None else None
        
        # Convert hand rotations (optional fields)
        left_hand_rot = np.array(data_dict['left_hand_rot'], dtype=np.float32) if data_dict.get('left_hand_rot') is not None else None
        right_hand_rot = np.array(data_dict['right_hand_rot'], dtype=np.float32) if data_dict.get('right_hand_rot') is not None else None
        
        # Create TeleData instance
        tele_data = TeleData(
            head_pose=head_pose,
            left_arm_pose=left_arm_pose,
            right_arm_pose=right_arm_pose,
            left_hand_pos=left_hand_pos,
            right_hand_pos=right_hand_pos,
            left_hand_rot=left_hand_rot,
            right_hand_rot=right_hand_rot,
            left_pinch_value=data_dict.get('left_pinch_value'),
            right_pinch_value=data_dict.get('right_pinch_value'),
            left_trigger_value=data_dict.get('left_trigger_value'),
            right_trigger_value=data_dict.get('right_trigger_value'),
            tele_state=tele_state
        )
        
        return tele_data
        
    def start(self):
        """Start the WebSocket client thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        logger_mp.info(f"WebSocket client started, connecting to {self.uri}")
        
    def stop(self):
        """Stop the WebSocket client thread"""
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5.0)
        logger_mp.info("WebSocket client stopped")
        
    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._receive_data())
        finally:
            self.loop.close()
            
    async def _receive_data(self):
        """Main async function to receive data from WebSocket"""
        reconnect_delay = 1.0
        max_reconnect_delay = 30.0
        
        while self.running:
            try:
                logger_mp.info(f"Attempting to connect to {self.uri}")
                
                # Configure WebSocket connection for low latency
                async with websockets.connect(
                    self.uri,
                    compression="deflate",  # Disable compression for lower latency
                    max_size=10 * 1024 * 1024,  # 10MB max message size
                    max_queue=1,  # Minimal queue
                    write_limit=0,  # No write buffer limit
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=60
                ) as websocket:
                    
                    logger_mp.info(f"Successfully connected to WebSocket server at {self.uri}")
                    reconnect_delay = 1.0  # Reset reconnect delay on successful connection
                    
                    # Create tasks for receiving and latency measurement
                    tasks = []
                    receive_task = asyncio.create_task(self._receive_loop(websocket))
                    tasks.append(receive_task)
                    
                    if self.measure_latency:
                        latency_task = asyncio.create_task(self._measure_latency(websocket))
                        tasks.append(latency_task)
                        
                    # Run until disconnected or stopped
                    try:
                        await asyncio.gather(*tasks)
                    except asyncio.CancelledError:
                        logger_mp.info("Tasks cancelled")
                        break
                        
            except websockets.exceptions.InvalidURI as e:
                logger_mp.error(f"Invalid WebSocket URI: {e}")
                break  # Don't retry on invalid URI
                
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.WebSocketException,
                    ConnectionRefusedError,
                    OSError) as e:
                logger_mp.warning(f"WebSocket connection failed: {e}")
                
            except Exception as e:
                logger_mp.error(f"Unexpected WebSocket error: {e}", exc_info=True)
                
            if self.running:
                logger_mp.info(f"Will reconnect in {reconnect_delay:.1f} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            else:
                logger_mp.info("Stopping reconnection attempts")
                break
                
    async def _receive_loop(self, websocket):
        """Dedicated loop for receiving messages"""
        while self.running:
            try:
                # Receive with short timeout
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                
                # Skip processing if we're not running anymore
                if not self.running:
                    break
                
                # Initialize data as None
                data = None
                
                # Check message type and process accordingly
                if isinstance(message, str):
                    # Handle string messages
                    if message == "pong":
                        continue  # Skip pong messages
                    else:
                        # Try to parse as JSON
                        try:
                            logger_mp.info("Received a message")
                            data = json.loads(message)
                        except json.JSONDecodeError as e:
                            logger_mp.debug(f"Received non-JSON string: {message[:100]}")
                            continue
                elif isinstance(message, bytes):
                    # Handle binary messages
                    try:
                        if self.use_msgpack:
                            data = msgpack.unpackb(message, raw=False)
                        else:
                            data = json.loads(message.decode('utf-8'))
                    except msgpack.exceptions.ExtraData as e:
                        logger_mp.error(f"Error parsing msgpack: {e}")
                        continue
                    except json.JSONDecodeError as e:
                        logger_mp.error(f"Error parsing JSON from bytes: {e}")
                        continue
                    except Exception as e:
                        logger_mp.error(f"Error deserializing message: {e}")
                        continue
                
                # If we didn't get valid data, continue
                if data is None:
                    continue
                    
                # Ensure data is a dictionary
                if not isinstance(data, dict):
                    logger_mp.warning(f"Received non-dictionary data: {type(data)}")
                    continue
                
                # Track latency if timestamp present
                if 'timestamp' in data:
                    latency = time.time() - data['timestamp']
                    with self.stats_lock:
                        self.latency_samples.append(latency)
                        
                # Track dropped frames
                if 'frame_id' in data:
                    frame_id = data['frame_id']
                    if self.last_frame_id >= 0:
                        dropped = frame_id - self.last_frame_id - 1
                        if dropped > 0:
                            with self.stats_lock:
                                self.dropped_frames += dropped
                    self.last_frame_id = frame_id
                    
                # Deserialize to TeleData dataclass
                try:
                    tele_data = self.deserialize_teledata(data)
                except Exception as e:
                    logger_mp.error(f"Error deserializing TeleData: {e}", exc_info=True)
                    continue
                
                # Store in queue (latest only)
                try:
                    # Clear queue and add new data
                    while True:
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            break
                            
                    self.data_queue.put_nowait(tele_data)
                except Exception as e:
                    logger_mp.error(f"Error updating queue: {e}")
                    
                # Update statistics
                with self.stats_lock:
                    self.receive_count += 1
                    if isinstance(message, bytes):
                        self.total_bytes_received += len(message)
                    else:
                        self.total_bytes_received += len(message.encode('utf-8'))
                        
            except asyncio.TimeoutError:
                # This is normal - just no data received within timeout
                continue
            except websockets.exceptions.ConnectionClosed as e:
                logger_mp.warning(f"WebSocket connection closed: {e}")
                break  # Exit loop to reconnect
            except Exception as e:
                logger_mp.error(f"Unexpected error in receive loop: {e}", exc_info=True)
                # Don't break on unexpected errors, try to continue
                await asyncio.sleep(0.1)
        
        logger_mp.info("Exiting receive loop")
                
    async def _measure_latency(self, websocket):
        """Periodically measure round-trip latency"""
        while self.running:
            try:
                start_time = time.time()
                await websocket.send("ping")
                pong = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                
                if pong == "pong":
                    latency = time.time() - start_time
                    # Report latency to server
                    await websocket.send(f"latency:{latency}")
                    
                await asyncio.sleep(5.0)  # Measure every 5 seconds
                
            except Exception:
                break
                
    def get_latest_data(self, timeout: float = 0.001) -> Optional[TeleData]:
        """Get only the most recent data, discard old"""
        try:
            # Get all available data but keep only the last
            data = None
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                
            # If nothing was available, wait briefly
            if data is None and timeout > 0:
                try:
                    data = self.data_queue.get(timeout=timeout)
                except queue.Empty:
                    pass
                    
            return data
            
        except Exception as e:
            return None

    def print_stats(self):
        """Print detailed performance statistics"""
        with self.stats_lock:
            current_time = time.time()
            time_diff = current_time - self.last_stats_time
            
            if time_diff > 0 and self.receive_count > 0:
                frequency = self.receive_count / time_diff
                bandwidth = self.total_bytes_received / time_diff / 1024
                avg_bytes = self.total_bytes_received / self.receive_count
                
                avg_latency = np.mean(self.latency_samples) * 1000 if self.latency_samples else 0
                max_latency = np.max(self.latency_samples) * 1000 if self.latency_samples else 0
                min_latency = np.min(self.latency_samples) * 1000 if self.latency_samples else 0
                
                print(f"\n--- WebSocket Client Stats ---")
                print(f"Receive frequency: {frequency:.1f} Hz")
                print(f"Bandwidth: {bandwidth:.1f} KB/s")
                print(f"Avg message size: {avg_bytes:.0f} bytes")
                print(f"Total received: {self.receive_count} messages")
                print(f"Dropped frames: {self.dropped_frames}")
                
                if self.latency_samples:
                    print(f"Latency - Avg: {avg_latency:.1f}ms, "
                          f"Min: {min_latency:.1f}ms, Max: {max_latency:.1f}ms")
                print("------------------------------\n")
                
                # Reset counters
                self.last_stats_time = current_time
                self.receive_count = 0
                self.total_bytes_received = 0
                self.latency_samples = []
                self.dropped_frames = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--frequency', type = float, default = 60.0, help = 'save data\'s frequency')

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording')
    parser.add_argument('--tast-name', type = str, default = 'pick cube', help = 'task name for recording')
    parser.add_argument('--task-goal', type = str, default = 'e.g. pick the red cube on the table.', help = 'task goal for recording')

    # remote flags
    parser.add_argument('--websocket_uri', type = str, default = 'ws://10.0.0.37:8765', help = 'WebSocket server URI')
    parser.add_argument('--image_server_ip', type = str, default = '127.0.0.1', help = 'Image server IP')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    if args.sim:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [540, 960],  # Head camera resolution
            'head_camera_id_numbers': [0]
        }
    else:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [540, 960],  # Head camera resolution
            'head_camera_id_numbers': [0]
        }


    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST and args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name, server_address=args.image_server_ip)
    elif WRIST and not args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name, server_address=args.image_server_ip)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, server_address=args.image_server_ip)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # receive motion states from websocket
    ws_receiver = WebSocketReceiver(uri=args.websocket_uri)
    ws_receiver.start()
    tele_data = None

    # Wait for initial data
    wait_time = 0
    while wait_time < 5.0:  # Wait up to 5 seconds for initial data
        tele_data = ws_receiver.get_latest_data(timeout=0.1)
        if tele_data:
            logger_mp.info("Received initial data from WebSocket")
            break
        time.sleep(0.1)
        wait_time += 0.1
    else:
        logger_mp.warning("Warning: No initial data received")

    # Stats setting
    stats_interval = 5.0  # Print stats every 5 seconds
    last_stats_print = time.time()
    frame_counter = 0

    # arm
    if args.arm == "G1_29":
        arm_ik = G1_29_ArmIK()
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
    elif args.arm == "G1_23":
        arm_ik = G1_23_ArmIK()
        arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
    elif args.arm == "H1_2":
        arm_ik = H1_2_ArmIK()
        arm_ctrl = H1_2_ArmController(simulation_mode=args.sim)
    elif args.arm == "H1":
        arm_ik = H1_ArmIK()
        arm_ctrl = H1_ArmController(simulation_mode=args.sim)

    # end-effector
    if args.ee == "dex3":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
    elif args.ee == "inspire1":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        hand_ctrl = Inspire_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    elif args.ee == "brainco":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    else:
        pass

    # simulation mode
    if args.sim:
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        from teleop.utils.sim_state_topic import start_sim_state_subscribe
        sim_state_subscriber = start_sim_state_subscribe()

    # controller + motion mode
    if args.xr_mode == "controller" and args.motion:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        sport_client = LocoClient()
        sport_client.SetTimeout(0.0001)
        sport_client.Init()
    
    # record + headless mode
    if args.record and args.headless:
        recorder = EpisodeWriter(task_dir = args.task_dir + args.tast_name, task_goal = args.task_goal, frequency = args.frequency, rerun_log = False)
    elif args.record and not args.headless:
        recorder = EpisodeWriter(task_dir = args.task_dir + args.tast_name, task_goal = args.task_goal, frequency = args.frequency, rerun_log = True)
        
    try:
        logger_mp.info("Please enter the start signal (enter 'c' to start the subsequent program)")
        while not start_signal:
            time.sleep(0.01)
        
        arm_ctrl.speed_gradual_max()

        # Stats for error tracking
        consecutive_errors = 0
        max_consecutive_errors = 10
        error_cooldown = 0.1
        
        while running:
            try:
                start_time = time.time()
                frame_counter += 1

                if not args.headless:
                    tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                    cv2.imshow("record image", tv_resized_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        stop_listening()
                        running = False
                        if args.sim:
                            publish_reset_category(2, reset_pose_publisher)
                    elif key == ord('s'):
                        should_toggle_recording = True
                    elif key == ord('a'):
                        if args.sim:
                            publish_reset_category(2, reset_pose_publisher)

                if args.record and should_toggle_recording:
                    should_toggle_recording = False
                    if not is_recording:
                        if recorder.create_episode():
                            is_recording = True
                        else:
                            logger_mp.error("Failed to create episode. Recording not started.")
                    else:
                        is_recording = False
                        recorder.save_episode()
                        if args.sim:
                            publish_reset_category(1, reset_pose_publisher)

                # get input data
                t_get_start = time.time()
                tele_data = ws_receiver.get_latest_data(timeout=0.001)
                t_get_end = time.time()

                if t_get_end - t_get_start > 0.01:  # Log if it takes more than 10ms
                    logger_mp.warning(f"get_latest_data took {t_get_end - t_get_start:.3f}s!")

                if tele_data is None:
                    # logger_mp.warning("No teledata received, skipping")
                    # time.sleep(0.01)
                    continue

                if (args.ee == "dex3" or args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                    with left_hand_pos_array.get_lock():
                        left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
                elif args.ee == "dex1" and args.xr_mode == "controller":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_trigger_value
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_trigger_value
                elif args.ee == "dex1" and args.xr_mode == "hand":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_pinch_value
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_pinch_value
                else:
                    pass        
                
                # high level control
                if args.xr_mode == "controller" and args.motion:
                    # quit teleoperate
                    if tele_data.tele_state.right_aButton:
                        stop_listening()
                        running = False
                    # command robot to enter damping mode. soft emergency stop function
                    if tele_data.tele_state.left_thumbstick_state and tele_data.tele_state.right_thumbstick_state:
                        sport_client.Damp()
                    # control, limit velocity to within 0.3
                    sport_client.Move(-tele_data.tele_state.left_thumbstick_value[1]  * 0.3,
                                    -tele_data.tele_state.left_thumbstick_value[0]  * 0.3,
                                    -tele_data.tele_state.right_thumbstick_value[0] * 0.3)

                # get current robot state data.
                t1 = time.time()
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
                t2 = time.time()

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_arm_pose, tele_data.right_arm_pose, current_lr_arm_q, current_lr_arm_dq)
                t3 = time.time()
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")

                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                t4 = time.time()

                # if frame_counter % 1 == 0:
                #     logger_mp.info(f"Timing - Get state: {t2-t1:.3f}s, IK: {t3-t2:.3f}s, Control: {t4-t3:.3f}s")

                # record data
                if args.record:
                    # dex hand or gripper
                    if args.ee == "dex3" and args.xr_mode == "hand":
                        with dual_hand_data_lock:
                            left_ee_state = dual_hand_state_array[:7]
                            right_ee_state = dual_hand_state_array[-7:]
                            left_hand_action = dual_hand_action_array[:7]
                            right_hand_action = dual_hand_action_array[-7:]
                            current_body_state = []
                            current_body_action = []
                    elif args.ee == "dex1" and args.xr_mode == "hand":
                        with dual_gripper_data_lock:
                            left_ee_state = [dual_gripper_state_array[0]]
                            right_ee_state = [dual_gripper_state_array[1]]
                            left_hand_action = [dual_gripper_action_array[0]]
                            right_hand_action = [dual_gripper_action_array[1]]
                            current_body_state = []
                            current_body_action = []
                    elif args.ee == "dex1" and args.xr_mode == "controller":
                        with dual_gripper_data_lock:
                            left_ee_state = [dual_gripper_state_array[0]]
                            right_ee_state = [dual_gripper_state_array[1]]
                            left_hand_action = [dual_gripper_action_array[0]]
                            right_hand_action = [dual_gripper_action_array[1]]
                            current_body_state = arm_ctrl.get_current_motor_q().tolist()
                            current_body_action = [-tele_data.tele_state.left_thumbstick_value[1]  * 0.3,
                                                -tele_data.tele_state.left_thumbstick_value[0]  * 0.3,
                                                -tele_data.tele_state.right_thumbstick_value[0] * 0.3]
                    elif (args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                        with dual_hand_data_lock:
                            left_ee_state = dual_hand_state_array[:6]
                            right_ee_state = dual_hand_state_array[-6:]
                            left_hand_action = dual_hand_action_array[:6]
                            right_hand_action = dual_hand_action_array[-6:]
                            current_body_state = []
                            current_body_action = []
                    else:
                        left_ee_state = []
                        right_ee_state = []
                        left_hand_action = []
                        right_hand_action = []
                        current_body_state = []
                        current_body_action = []
                    # head image
                    current_tv_image = tv_img_array.copy()
                    # wrist image
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()
                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]
                    if is_recording:
                        colors = {}
                        depths = {}
                        if BINOCULAR:
                            colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                            colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        else:
                            colors[f"color_{0}"] = current_tv_image
                            if WRIST:
                                colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        states = {
                            "left_arm": {                                                                    
                                "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                                "qvel":   [],                          
                                "torque": [],                        
                            }, 
                            "right_arm": {                                                                    
                                "qpos":   right_arm_state.tolist(),       
                                "qvel":   [],                          
                                "torque": [],                         
                            },                        
                            "left_ee": {                                                                    
                                "qpos":   left_ee_state,           
                                "qvel":   [],                           
                                "torque": [],                          
                            }, 
                            "right_ee": {                                                                    
                                "qpos":   right_ee_state,       
                                "qvel":   [],                           
                                "torque": [],  
                            }, 
                            "body": {
                                "qpos": current_body_state,
                            }, 
                        }
                        actions = {
                            "left_arm": {                                   
                                "qpos":   left_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],      
                            }, 
                            "right_arm": {                                   
                                "qpos":   right_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],       
                            },                         
                            "left_ee": {                                   
                                "qpos":   left_hand_action,       
                                "qvel":   [],       
                                "torque": [],       
                            }, 
                            "right_ee": {                                   
                                "qpos":   right_hand_action,       
                                "qvel":   [],       
                                "torque": [], 
                            }, 
                            "body": {
                                "qpos": current_body_action,
                            }, 
                        }
                        if args.sim:
                            sim_state = sim_state_subscriber.read_data()            
                            recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                        else:
                            recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
                logger_mp.debug(f"main process sleep: {sleep_time}")

                # Print statistics periodically
                if current_time - last_stats_print >= stats_interval:
                    ws_receiver.print_stats()
                    
                    # Also print your main loop stats
                    loop_frequency = frame_counter / stats_interval
                    logger_mp.info(f"Main loop frequency: {loop_frequency:.1f} Hz")
                    
                    # Reset counters
                    last_stats_print = current_time
                    frame_counter = 0
                    
            except Exception as e:
                # Catch any unexpected errors in the main loop
                consecutive_errors += 1
                logger_mp.error(f"Main loop error ({consecutive_errors}/{max_consecutive_errors}): {e}", exc_info=True)
                
                if consecutive_errors >= max_consecutive_errors:
                    logger_mp.critical(f"Too many consecutive errors ({consecutive_errors}), exiting")
                    running = False
                    break
                    
                # Cool down before retrying
                time.sleep(error_cooldown)
                continue

    except Exception as e:
        logger_mp.error(f"Error while running: {e}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")

    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        if args.sim:
            sim_state_subscriber.stop_subscribe()
        tv_img_shm.close()
        tv_img_shm.unlink()
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()
        if args.record:
            recorder.close()
        listen_keyboard_thread.join()
        logger_mp.info("Finally, exiting program...")
        exit(0)
