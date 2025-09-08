import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import threading
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.robot_hand_brainco import Brainco_Controller

# from teleop.image_server.image_client import ImageClient
from teleop.image_server.image_client_fast import ImageClient

from teleop.utils.episode_writer import EpisodeWriter
from sshkeyboard import listen_keyboard, stop_listening

# for remote operation
import asyncio
import websockets
import json
import msgpack
import queue
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, asdict
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
    if key == 'r':
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

class WebSocketServer:
    """High-performance WebSocket server for TeleData transmission"""
    
    def __init__(self, host="0.0.0.0", port=8765, use_msgpack=True, logger=None):
        self.host = host
        self.port = port
        self.use_msgpack = use_msgpack
        
        # Use a single queue with latest-only strategy
        self.data_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.connected_clients: Set = set()
        
        # Performance tracking
        self.stats_lock = threading.Lock()
        self.send_count = 0
        self.last_stats_time = time.time()
        self.last_stats_count = 0
        self.total_bytes_sent = 0
        self.latency_samples = []
        
    def serialize_teledata(self, tele_data: TeleData) -> Dict[str, Any]:
        """Efficiently serialize TeleData to dictionary format"""
        serialized = {
            'timestamp': time.time(),
            'frame_id': self.send_count,
        }
        
        # Serialize numpy arrays and handle None values
        # Head, arm poses (required fields)
        serialized['head_pose'] = tele_data.head_pose.tolist() if tele_data.head_pose is not None else None
        serialized['left_arm_pose'] = tele_data.left_arm_pose.tolist() if tele_data.left_arm_pose is not None else None
        serialized['right_arm_pose'] = tele_data.right_arm_pose.tolist() if tele_data.right_arm_pose is not None else None
        
        # Hand positions (optional)
        serialized['left_hand_pos'] = tele_data.left_hand_pos.tolist() if tele_data.left_hand_pos is not None else None
        serialized['right_hand_pos'] = tele_data.right_hand_pos.tolist() if tele_data.right_hand_pos is not None else None
        
        # Hand rotations (optional)
        serialized['left_hand_rot'] = tele_data.left_hand_rot.tolist() if tele_data.left_hand_rot is not None else None
        serialized['right_hand_rot'] = tele_data.right_hand_rot.tolist() if tele_data.right_hand_rot is not None else None
        
        # Scalar values
        serialized['left_pinch_value'] = tele_data.left_pinch_value
        serialized['right_pinch_value'] = tele_data.right_pinch_value
        serialized['left_trigger_value'] = tele_data.left_trigger_value
        serialized['right_trigger_value'] = tele_data.right_trigger_value
        
        # Serialize TeleStateData
        if tele_data.tele_state is not None:
            state_dict = {}
            state = tele_data.tele_state
            
            # Boolean and scalar fields
            for field_name in ['left_pinch_state', 'right_pinch_state',
                              'left_squeeze_state', 'right_squeeze_state',
                              'left_squeeze_value', 'right_squeeze_value',
                              'left_trigger_state', 'right_trigger_state',
                              'left_squeeze_ctrl_state', 'right_squeeze_ctrl_state',
                              'left_squeeze_ctrl_value', 'right_squeeze_ctrl_value',
                              'left_thumbstick_state', 'right_thumbstick_state',
                              'left_aButton', 'right_aButton',
                              'left_bButton', 'right_bButton']:
                state_dict[field_name] = getattr(state, field_name, None)
            
            # Numpy array fields
            if hasattr(state, 'left_thumbstick_value') and state.left_thumbstick_value is not None:
                state_dict['left_thumbstick_value'] = state.left_thumbstick_value.tolist()
            else:
                state_dict['left_thumbstick_value'] = None
                
            if hasattr(state, 'right_thumbstick_value') and state.right_thumbstick_value is not None:
                state_dict['right_thumbstick_value'] = state.right_thumbstick_value.tolist()
            else:
                state_dict['right_thumbstick_value'] = None
                
            serialized['tele_state'] = state_dict
        else:
            serialized['tele_state'] = None
        
        return serialized
        
    def start(self):
        """Start the WebSocket server in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger_mp.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        logger_mp.info(f"Using: {'msgpack' if self.use_msgpack else 'JSON'} serialization")
        
    def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            
    def send_data(self, tele_data: TeleData):
        """Add TeleData to queue (overwrites previous if not sent)"""
        try:
            # Serialize immediately to avoid holding references
            serialized = self.serialize_teledata(tele_data)
            
            # Always use latest data - clear queue and add new
            try:
                while True:
                    self.data_queue.get_nowait()
            except queue.Empty:
                pass
                
            self.data_queue.put_nowait(serialized)
        except Exception as e:
            logger_mp.error(f"Error queuing data: {e}")
            
    def print_stats(self):
        """Print detailed performance statistics"""
        with self.stats_lock:
            current_time = time.time()
            time_diff = current_time - self.last_stats_time
            count_diff = self.send_count - self.last_stats_count
            
            if time_diff > 0:
                frequency = count_diff / time_diff
                avg_bytes = self.total_bytes_sent / max(self.send_count, 1)
                bandwidth = (self.total_bytes_sent - self.last_stats_count * avg_bytes) / time_diff / 1024
                
                avg_latency = np.mean(self.latency_samples) * 1000 if self.latency_samples else 0
                
                print(f"\n--- WebSocket Performance ---")
                print(f"Send frequency: {frequency:.1f} Hz")
                print(f"Bandwidth: {bandwidth:.1f} KB/s")
                print(f"Avg message size: {avg_bytes:.0f} bytes")
                print(f"Connected clients: {len(self.connected_clients)}")
                print(f"Queue size: {self.data_queue.qsize()}")
                if self.latency_samples:
                    print(f"Avg round-trip latency: {avg_latency:.1f} ms")
                print("-----------------------------\n")
                
                self.last_stats_time = current_time
                self.last_stats_count = self.send_count
                self.latency_samples = []
                
    def _run_server(self):
        """Run the async server in a new event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_main())
        finally:
            loop.close()
            
    async def _async_main(self):
        """Main async function to start server and run broadcast"""
        # Configure WebSocket settings for low latency
        server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            compression=None,  # Disable compression for lower latency
            max_size=10 * 1024 * 1024,  # 10MB max message size
            max_queue=1,  # Minimal queue for real-time
            write_limit=0,  # No write buffer limit
            ping_interval=20,
            ping_timeout=10
        )
        
        logger_mp.info(f"Server ready for connections")
        
        # Create broadcast task
        broadcast_task = asyncio.create_task(self._broadcast_data())
        
        try:
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
        finally:
            broadcast_task.cancel()
            server.close()
            await server.wait_closed()
            
    async def _handle_client(self, websocket):
        """Handle a new client connection"""
        self.connected_clients.add(websocket)
        client_address = websocket.remote_address
        logger_mp.info(f"Client connected from {client_address}")
        
        try:
            # Handle ping messages for latency measurement
            async for message in websocket:
                if message == "ping":
                    await websocket.send("pong")
                elif message.startswith("latency:"):
                    # Client reporting latency
                    try:
                        latency = float(message.split(":")[1])
                        with self.stats_lock:
                            self.latency_samples.append(latency)
                    except:
                        pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)
            logger_mp.info(f"Client disconnected from {client_address}")
            
    async def _broadcast_data(self):
        """High-performance broadcast loop"""
        while self.running:
            try:
                # Get latest data with minimal timeout
                try:
                    data = self.data_queue.get(timeout=0.001)
                except queue.Empty:
                    await asyncio.sleep(0.001)
                    continue
                    
                if not self.connected_clients:
                    await asyncio.sleep(0.01)
                    continue
                    
                # Serialize message once for all clients
                if self.use_msgpack:
                    message = msgpack.packb(data, use_bin_type=True)
                else:
                    message = json.dumps(data).encode('utf-8')
                    
                # Send to all connected clients concurrently
                disconnected = set()
                send_tasks = []
                
                for client in self.connected_clients:
                    send_tasks.append(self._send_to_client(client, message, disconnected))
                    
                # Wait for all sends to complete
                if send_tasks:
                    await asyncio.gather(*send_tasks, return_exceptions=True)
                    
                # Update statistics
                with self.stats_lock:
                    successful_sends = len(self.connected_clients) - len(disconnected)
                    self.send_count += successful_sends
                    self.total_bytes_sent += len(message) * successful_sends
                    
                # Remove disconnected clients
                self.connected_clients -= disconnected
                
            except Exception as e:
                logger_mp.error(f"Broadcast error: {e}")
                await asyncio.sleep(0.001)
                
    async def _send_to_client(self, client, message, disconnected_set):
        """Send message to a single client with error handling"""
        try:
            await asyncio.wait_for(client.send(message), timeout=0.1)
        except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
            disconnected_set.add(client)
        except Exception as e:
            logger_mp.error(f"Error sending to client: {e}")
            disconnected_set.add(client)

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

    parser.add_argument('--websocket_port', type = int, default = 8765, help = 'WebSocket server port')
    parser.add_argument('--send_latest', action = 'store_true', default = True, help = 'Send only latest data (low latency mode)')
    parser.add_argument('--send_sequential', dest = 'send_latest', action = 'store_false', help = 'Send all data sequentially')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    if args.sim:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 640],  # Head camera resolution
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
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name, server_address="127.0.0.1")
    elif WRIST and not args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape, img_shm_name=tv_img_shm.name, 
                                 return_state_data=True, return_hand_rot_data = False)

    # Initialize WebSocket server
    ws_server = WebSocketServer(host="0.0.0.0", port=args.websocket_port)
    ws_server.start()
    
    # simulation mode
    if args.sim:
        ChannelFactoryInitialize(1)
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
        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        while not start_signal:
            time.sleep(0.01)

        # Stats printing variables
        stats_interval = 5.0  # Print stats every 5 seconds
        last_stats_print = time.time()
        
        while running:

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
            tele_data = tv_wrapper.get_motion_state_data()
            logger_mp.debug(tele_data)

            start_time = time.time()

            # Send data via WebSocket to all connected clients
            ws_server.send_data(tele_data)

            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_print >= stats_interval:
                ws_server.print_stats()
                last_stats_print = current_time

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    finally:
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
