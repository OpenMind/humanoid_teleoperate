# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, shared_memory, Array, Value, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, left_hand_array_in, right_hand_array_in, dual_hand_data_lock = None, dual_hand_state_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array_in, right_hand_array_in,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array_out, dual_hand_action_array_out))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array_in, right_hand_array_in, left_hand_state_array, right_hand_state_array,
                            dual_hand_data_lock = None, dual_hand_state_array_out = None, dual_hand_action_array_out = None):
        self.running = True

        left_q_target  = np.full(Dex3_Num_Motors, 0.0)
        right_q_target = np.full(Dex3_Num_Motors, 0.0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # Logging configuration
        LOG_INTERVAL = 1.0  # Log every 1 second (adjust as needed)
        last_log_time = 0
        iteration_count = 0
        
        # For computing running statistics
        left_errors_history = []
        right_errors_history = []
        MAX_HISTORY = 100  # Keep last 100 samples for statistics

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            # logger_mp.info(f"left: {id}")
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            # logger_mp.info(f"right: {id}")
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        try:
            while self.running:
                start_time = time.time()
                iteration_count += 1
                
                # get dual hand state
                with left_hand_array_in.get_lock():
                    left_hand_data  = np.array(left_hand_array_in[:]).reshape(25, 3).copy()
                with right_hand_array_in.get_lock():
                    right_hand_data = np.array(right_hand_array_in[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                left_q_state = np.array(left_hand_state_array[:])
                right_q_state = np.array(right_hand_state_array[:])
                state_data = np.concatenate((left_q_state, right_q_state))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # Apply offsets
                    # left_q_target[0] = left_q_target[0] + 0.3
                    # right_q_target[0] = right_q_target[0] + 0.3
                    left_q_target[0] = left_q_target[0] * 0.5 + 0.55 * 0.5
                    left_q_target[1] = left_q_target[1] * 0.6 + 0.5 * 0.4
                    left_q_target[2] = left_q_target[2] * 0.75 + 0.1 * 0.25
                    right_q_target[0] = right_q_target[0] * 0.5 + 0.55 * 0.5
                
                # logger_mp.info(f"set 0.7, left_q_target is now: {left_q_target}")
                # Compute errors
                left_errors = left_q_target - left_q_state
                right_errors = right_q_target - right_q_state
                
                # Store errors for statistics
                left_errors_history.append(left_errors)
                right_errors_history.append(right_errors)
                if len(left_errors_history) > MAX_HISTORY:
                    left_errors_history.pop(0)
                    right_errors_history.pop(0)
                
                # Log at intervals
                current_time = time.time()
                # if current_time - last_log_time >= LOG_INTERVAL:
                #     logger_mp.info("\n" + "="*80)
                #     logger_mp.info(f"[Iteration {iteration_count}] Control Loop Status Report")
                #     logger_mp.info("="*80)
                    
                #     # Log joint names for reference
                #     logger_mp.info("\nJoint Names:")
                #     logger_mp.info(f"  Left:  {self.hand_retargeting.left_dex3_api_joint_names}")
                #     logger_mp.info(f"  Right: {self.hand_retargeting.right_dex3_api_joint_names}")
                    
                #     # Log targets
                #     logger_mp.info("\nTarget Q values:")
                #     logger_mp.info(f"  Left:  {np.round(left_q_target, 3).tolist()}")
                #     logger_mp.info(f"  Right: {np.round(right_q_target, 3).tolist()}")
                    
                #     # Log current states
                #     logger_mp.info("\nCurrent Q states:")
                #     logger_mp.info(f"  Left:  {np.round(left_q_state, 3).tolist()}")
                #     logger_mp.info(f"  Right: {np.round(right_q_state, 3).tolist()}")
                    
                #     # Log errors
                #     logger_mp.info("\nPosition Errors (target - state):")
                #     logger_mp.info(f"  Left:  {np.round(left_errors, 3).tolist()}")
                #     logger_mp.info(f"  Right: {np.round(right_errors, 3).tolist()}")
                    
                #     # Log error statistics
                #     if len(left_errors_history) > 0:
                #         left_errors_array = np.array(left_errors_history)
                #         right_errors_array = np.array(right_errors_history)
                        
                #         logger_mp.info("\nError Statistics (over last {} samples):".format(len(left_errors_history)))
                #         logger_mp.info("  Left hand:")
                #         logger_mp.info(f"    Mean abs error: {np.round(np.mean(np.abs(left_errors_array), axis=0), 3).tolist()}")
                #         logger_mp.info(f"    Max abs error:  {np.round(np.max(np.abs(left_errors_array), axis=0), 3).tolist()}")
                #         logger_mp.info(f"    RMS error:      {np.round(np.sqrt(np.mean(left_errors_array**2, axis=0)), 3).tolist()}")
                        
                #         logger_mp.info("  Right hand:")
                #         logger_mp.info(f"    Mean abs error: {np.round(np.mean(np.abs(right_errors_array), axis=0), 3).tolist()}")
                #         logger_mp.info(f"    Max abs error:  {np.round(np.max(np.abs(right_errors_array), axis=0), 3).tolist()}")
                #         logger_mp.info(f"    RMS error:      {np.round(np.sqrt(np.mean(right_errors_array**2, axis=0)), 3).tolist()}")
                    
                #     # Log thumb values specifically (since that's your concern)
                #     logger_mp.info("\nThumb Analysis (first 3 joints):")
                #     logger_mp.info(f"  Left thumb:")
                #     logger_mp.info(f"    Target:  {np.round(left_q_target[:3], 3).tolist()}")
                #     logger_mp.info(f"    State:   {np.round(left_q_state[:3], 3).tolist()}")
                #     logger_mp.info(f"    Error:   {np.round(left_errors[:3], 3).tolist()}")
                #     logger_mp.info(f"  Right thumb:")
                #     logger_mp.info(f"    Target:  {np.round(right_q_target[:3], 3).tolist()}")
                #     logger_mp.info(f"    State:   {np.round(right_q_state[:3], 3).tolist()}")
                #     logger_mp.info(f"    Error:   {np.round(right_errors[:3], 3).tolist()}")
                    
                #     # Log hand keypoint data for debugging
                #     logger_mp.info("\nHand Keypoint Debug:")
                #     logger_mp.info(f"  Left hand keypoints used:")
                #     logger_mp.info(f"    Wrist (0):     {np.round(left_hand_data[0], 3).tolist()}")
                #     logger_mp.info(f"    Thumb tip (4): {np.round(left_hand_data[4], 3).tolist()}")
                #     logger_mp.info(f"    Index tip (9): {np.round(left_hand_data[9], 3).tolist()}")
                #     logger_mp.info(f"    Middle tip (14): {np.round(left_hand_data[14], 3).tolist()}")
                    
                #     logger_mp.info("\n" + "="*80 + "\n")
                    
                #     last_log_time = current_time
                
                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data

                # Only log basic info at high frequency (comment out if too verbose)
                # logger_mp.debug(f"left: {left_q_target}")
                # logger_mp.debug(f"right: {right_q_target}")
                # logger_mp.info(f"About to send to robot - left_q_target: {left_q_target}")
                self.ctrl_dual_hand(left_q_target, right_q_target)
                
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex3_1_Controller has been closed.")

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"

class Dex1_1_Gripper_Controller:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = True, fps = 200.0, Unit_Test = False, simulation_mode = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_gripper_value_in: [input] Left ctrl data (required from XR device) to control_thread

        right_gripper_value_in: [input] Right ctrl data (required from XR device) to control_thread

        dual_gripper_data_lock: Data synchronization lock for dual_gripper_state_array and dual_gripper_action_array

        dual_gripper_state_out: [output] Return left(1), right(1) gripper motor state

        dual_gripper_action_out: [output] Return left(1), right(1) gripper motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """

        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.simulation_mode = simulation_mode
        
        if filter:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)
 
        # initialize handcmd publisher and handstate subscriber
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        # Shared Arrays for gripper states
        self.left_gripper_state_value = Value('d', 0.0, lock=True)
        self.right_gripper_state_value = Value('d', 0.0, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while not self.gripper_sub_ready:
            time.sleep(0.01)
            logger_mp.warning("[Dex1_1_Gripper_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_gripper_value_in, right_gripper_value_in, self.left_gripper_state_value, self.right_gripper_state_value,
                                                                                         dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!\n")

    def _subscribe_gripper_state(self):
        while True:
            left_gripper_msg  = self.LeftGripperState_subscriber.Read()
            right_gripper_msg  = self.RightGripperState_subscriber.Read()
            self.gripper_sub_ready = True
            if left_gripper_msg is not None and right_gripper_msg is not None:
                self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                self.right_gripper_state_value.value = right_gripper_msg.states[0].q
            time.sleep(0.002)
    
    def ctrl_dual_gripper(self, dual_gripper_action):
        """set current left, right gripper motor cmd target q"""
        self.left_gripper_msg.cmds[0].q  = dual_gripper_action[0]
        self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
        # logger_mp.debug("gripper ctrl publish ok.")
    
    def control_thread(self, left_gripper_value_in, right_gripper_value_in, left_gripper_state_value, right_gripper_state_value, dual_hand_data_lock = None, 
                             dual_gripper_state_out = None, dual_gripper_action_out = None):
        self.running = True
        if self.simulation_mode:
            DELTA_GRIPPER_CMD = 0.18
        else:   
            DELTA_GRIPPER_CMD = 0.18     # The motor rotates 5.4 radians, the clamping jaw slide open 9 cm, so 0.6 rad <==> 1 cm, 0.18 rad <==> 3 mm
        THUMB_INDEX_DISTANCE_MIN = 5.0
        THUMB_INDEX_DISTANCE_MAX = 7.0
        LEFT_MAPPED_MIN  = 0.0           # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0           # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40 
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # initialize gripper cmd msg
        self.left_gripper_msg  = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq  = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp  = kp
        self.left_gripper_msg.cmds[0].kd  = kd

        self.right_gripper_msg.cmds[0].dq  = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp  = kp
        self.right_gripper_msg.cmds[0].kd  = kd
        try:
            while self.running:
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                with left_gripper_value_in.get_lock():
                    left_gripper_value  = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value
                # get current dual gripper motor state
                dual_gripper_state = np.array([left_gripper_state_value.value, right_gripper_state_value.value])
                
                if left_gripper_value != 0.0 or right_gripper_value != 0.0: # if input data has been initialized.
                    # Linear mapping from [0, THUMB_INDEX_DISTANCE_MAX] to gripper action range
                    left_target_action  = np.interp(left_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                    right_target_action = np.interp(right_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])
                # clip dual gripper action to avoid overflow
                if not self.simulation_mode:
                    left_actual_action  = np.clip(left_target_action,  dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD) 
                    right_actual_action = np.clip(right_target_action, dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD)
                else:
                    left_actual_action  = left_target_action
                    right_actual_action = right_target_action
                dual_gripper_action = np.array([left_actual_action, right_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])

                self.ctrl_dual_gripper(dual_gripper_action)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex1_1_Gripper_Controller has been closed.")

class Gripper_JointIndex(IntEnum):
    kGripper = 0


if __name__ == "__main__":
    import argparse
    from televuer import TeleVuerWrapper
    from teleop.image_server.image_client import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    args = parser.parse_args()
    logger_mp.info(f"args:{args}\n")

    # image
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [0],
    }
    ASPECT_RATIO_THRESHOLD = 2.0  # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    # image
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)
    img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)
    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape, img_shm_name=tv_img_shm.name, 
                                 return_state_data=True, return_hand_rot_data = False)

# end-effector
    if args.ee == "dex3":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        while True:
            tele_data = tv_wrapper.get_motion_state_data()
            if args.ee == "dex3" and args.xr_mode == "hand":
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

            # with dual_hand_data_lock:
            #     logger_mp.info(f"state : {list(dual_hand_state_array)} \naction: {list(dual_hand_action_array)} \n")
            time.sleep(0.01)
