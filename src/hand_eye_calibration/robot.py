"""
kinova.py

An implementation of the Robot class with the Kinova robotic arm.
"""

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override
import time
import threading
from roboenv import Robot

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient

from kortex_api.autogen.messages import Session_pb2, Base_pb2, VisionConfig_pb2, DeviceConfig_pb2
from kortex_api.Exceptions.KException import KException

class Kinova(Robot):
    """
    A Robot object representing a Kinova robotic arm.
    """

    # TIMEOUT = 20 # seconds; max waiting time during actions

    def __init__(self, fps: int, id: int) -> None:
        """
        Initialize a Kinova robot instance.
        
        Does not establish hardware connection--use launch() for that.

        N.B. Some of these attributes are not used/needed.

        Args:
            fps (int): frames per second
            id (int): unique identifier for robot
        """
        
        super().__init__(fps)
        self.id= id
        self.ip = "192.168.1.10"
        self.state = None
        self.bounds: list[tuple[float, float]] = [(0.15, 0.75), (-0.60, 0.42), (0.02, 0.40)]
        """operating area bounds in (x,y,z)"""
        self.height: float = 0.30
        """default navigation height (m)"""
        
        self.connected = False
        self.busy = False
        # self.nb_dof = 0
        self.action = Base_pb2.Action()

        self.gripper_command = Base_pb2.GripperCommand()
        self.gripper_command.mode = Base_pb2.GRIPPER_POSITION
        self.gripper_request = Base_pb2.GripperRequest()
        self.gripper_request.mode = Base_pb2.GRIPPER_POSITION

        self.notification_handles = []
        self.e = None

        self.error_callback = lambda kException: print("_________ callback error _________ {}".format(kException))
        self.transport = None
        self.router = None
        self.session_manager = None
        self.device_config_client = None
        self.base = None
        self.base_cyclic = None
        self.control_config_client = None

        # self.TWIST_TIME = 0.01

    @override
    def launch(self) -> None:
        """
        Initializes a robot connection.
        
        - Establishes connection to Kinova robot (TCP)
        - Initiates a session; SessionManager manages connection to robot base and ensures that robot can respond to commands.
        - Sets up communication clients; RouterClient allows for commands to be sent to robot base.
        - More info here: https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/blob/master/linked_md/python_transport_router_session_notif.md
        - Additionally, sets servoing_mode to SINGLE_LEVEL_SERVOING (for high-level movement).
        """
        
        TCP_PORT = 10000
        # UDP_PORT = 10001

        if self.connected:
            self.shutdown()

        try:
            self.transport = TCPTransport()
            self.router = RouterClient(self.transport, self.error_callback)
            self.transport.connect(self.ip, TCP_PORT)

            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = "admin"
            session_info.password = "admin"
            session_info.session_inactivity_timeout = 60000  # milliseconds
            session_info.connection_inactivity_timeout = 2000  # milliseconds

            self.session_manager = SessionManager(self.router)
            self.session_manager.CreateSession(session_info)

            print("Session created")

            self.base = BaseClient(self.router)
            self.base_cyclic = BaseCyclicClient(self.router)
            # self.device_config_client = DeviceConfigClient(self.router)
            # self.control_config_client = BaseClient(self.router)

            # self.nb_dof = self.base.GetActuatorCount().count

            servoing_mode = Base_pb2.ServoingModeInformation()
            servoing_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(servoing_mode)

            self.connected = True

        except KException as ex:
            self.on_error(ex)

    @override
    def shutdown(self) -> None:
        """
        Safely disconnect robot.

        Terminates active session and client objects.
        """
        
        # if self.session_manager != None:
        if self.connected:
            # self.session_manager.CloseSession()
            # self.router.SetActivationStatus(False)
            # self.transport.disconnect()
            # # Implementation in example code:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.session_manager.CloseSession(router_options)
            self.transport.disconnect()

        self.connected = False
        self.base = None
        # self.device_config_client = None
        # self.control_config_client = None
        self.session_manager = None
        self.router = None
        self.transport = None


    @override
    def get_current_state(self, joints=False):
        """
        Query the robot's current state.

        Args:
            joints (bool): if True, return joint angles instead of end-effector pose
        Returns:
            KinovaState: current state (end-effector pose & gripper position)
        """
        
        feedback = self.base_cyclic.RefreshFeedback()

        proprio: NDArray = np.array(
            [
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
                feedback.base.tool_pose_theta_x,
                feedback.base.tool_pose_theta_y,
                feedback.base.tool_pose_theta_z,
            ]
        )

        gripper_measure = self.base.GetMeasuredGripperMovement(self.gripper_request)
        gripper: float = gripper_measure.finger[0].value
        
        if joints:
            q = [feedback.actuators[i].position for i in range(7)]  # radians
            return q

        self.state = KinovaState(proprio, gripper)
        return KinovaState(proprio, gripper)

    @override
    def execute_action(self, action) -> None:
        """
        Execute a control action on the robot.

        Twist command specifies velocities; m/s and deg/s for our purposes.

        Args:
            action (KinovaAction): The action (twist command & gripper position) to send.
        """

        if action.gripper > 1.0 or action.gripper < 0.0:
            print("invalid gripper command")
            return
        
        command = Base_pb2.TwistCommand()

        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = 0  # duration constraint
        twist = command.twist

        dt = 1.0 / float(self.fps if getattr(self, "fps", 0) else 10.0)
        eps = 0.001 # small tolerance
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds

        def limit_axis(pos, vel, lo, hi):
            next_pos = pos + vel * dt

            # If we're inside, block steps that would leave the box this tick
            if lo + eps <= pos <= hi - eps:
                if next_pos < lo or next_pos > hi:
                    return 0.0
                return vel

            # If we're at/beyond a boundary, only allow velocity pointing back in
            if pos <= lo + eps:
                print("[SAFETY] boundary reached")
                return max(0.0, vel) 
            if pos >= hi - eps:
                print("[SAFETY] boundary reached")
                return min(0.0, vel)  
            return vel
        
        x, y, z = self.get_current_state().data[:3]
        vx, vy, vz = action.data[:3]
        vx = limit_axis(x, vx, xmin, xmax)
        vy = limit_axis(y, vy, ymin, ymax)
        vz = limit_axis(z, vz, zmin, zmax)

        twist.linear_x = vx
        twist.linear_y = vy
        twist.linear_z = vz
        twist.angular_x = action.data[3]  # should be deg/s
        twist.angular_y = action.data[4]
        twist.angular_z = action.data[5]

        # # I think twist should be updated continuously as-is, but in case time is needed to execute the command:
        # time.sleep(self.TWIST_TIME)

        # I think time.sleep(seconds) is the amount of time that the TwistCommand is to be executed for
        # then you stop the movement with 
        # base.Stop()

        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = action.gripper

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.notification_handles.append(notification_handle)

        self.base.SendTwistCommand(command)
        self.base.SendGripperCommand(gripper_command)
        e.wait(20) 
        self.base.Unsubscribe(notification_handle)

    @override
    def go_to_waypoint(self, waypoint) -> None:
        """
        Command the robot to move to a specific Cartesian waypoint.
        
        Does not change end-effector orientation (theta_x, theta_y, theta_z).

        Args:
            waypoint (Waypoint): The target waypoint (x, y, z).
        """

        self.action.Clear()
        self.action = Base_pb2.Action()
        self.action.name = "go_to_waypoint"
        self.action.application_data = ""

        feedback = self.base_cyclic.RefreshFeedback()
        pose = self.action.reach_pose.target_pose

        # check if the robot is in bounds
        if not self.in_bounds(waypoint):
            pose.x = feedback.base.tool_pose_x
            pose.y = feedback.base.tool_pose_y
            pose.z = feedback.base.tool_pose_z
            pose.theta_x = feedback.base.tool_pose_theta_x
            pose.theta_y = feedback.base.tool_pose_theta_y
            pose.theta_z = feedback.base.tool_pose_theta_z
            print("Waypoint is out of bounds, keeping current position")
        else:
            pose.x = float(waypoint.data[0])
            pose.y = float(waypoint.data[1])
            pose.z = float(waypoint.data[2])
            pose.theta_x = feedback.base.tool_pose_theta_x
            pose.theta_y = feedback.base.tool_pose_theta_y
            pose.theta_z = feedback.base.tool_pose_theta_z

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.notification_handles.append(notification_handle)

        self.base.ExecuteAction(self.action)
        e.wait(20)  # wait for up to 20 seconds or until action is completed
        self.base.Unsubscribe(notification_handle)


    # Note: Waypoint is defined in ../agent/aria.py
    def in_bounds(self, waypoint = None, action = None) -> bool:
        """
        Returns True if the robot is (or will remain) within the operation area.

        Args:
            waypoint (Waypoint, optional): If provided, check this pose instead of current state.
            action (KinovaAction, optional): If provided, predict the next pose by applying
                                            the action's velocity for one control step (dt=1/fps).
        """
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds

        # --- base pose to evaluate ---
        if waypoint is not None:
            x, y, z = waypoint.data[:3]
        else:
            x, y, z = self.get_current_state().data[:3]

        # --- project forward if action is given ---
        if action is not None:
            dt = 1.0 / float(self.fps if getattr(self, "fps", 0) else 10.0)
            vx, vy, vz = action.data[:3]  # linear velocities in m/s
            x += vx * dt
            y += vy * dt
            z += vz * dt

        return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax

    @staticmethod
    def check_for_end_or_abort(e):
        """
        Return a closure checking for END or ABORT notifications.
            
        Used in execute_action and go_to_waypoint.

        Args:
            e (threading.Event): event to signal when the action is completed; will be set when an END or ABORT occurs

        Returns:
            check: callback function to handle notifications; sets event when action ends or is aborted
        """

        def check(notification, e=e):
            # print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()

        return check

    def execute_seq(self) -> None:
        """
        Execute a predefined sequence.
        Uses predefined "test seq" sequence in Kinova.
        """
    
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.notification_handles.append(notification_handle)
        
        sequence_type = Base_pb2.RequestedActionType()
        sequence_type.action_type = Base_pb2.EXECUTE_SEQUENCE
        sequence_list = self.base.ReadAllSequences()
        for seq in sequence_list.sequence_list:
            if seq.name == "test seq":
                sequence_handle = seq.handle
        
        self.base.PlaySequence(sequence_handle)

        finished = e.wait(10)
        self.base.Unsubscribe(notification_handle)


    def go_home(self) -> None:
        """
        Go to designated home waypoint.
        Uses predefined "home_sept" action in Kinova.
        """

        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "home_sept":
                action_handle = action.handle
        
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = 0.0

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.notification_handles.append(notification_handle)

        self.base.SendGripperCommand(gripper_command)
        self.base.ExecuteActionFromReference(action_handle)

        finished = e.wait(10)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Home position reached")
        else:
            print("Could not reach position in time (10 seconds right now)")
        
    def get_eye_intrinsics(self):
        """
        Fetch intrinsics and extrinsics.
        """
        device_manager = DeviceManagerClient(self.router)
        device_handles = device_manager.ReadAllDevices()
        vision_device_ids = [
            handle.device_identifier for handle in device_handles.device_handle
            if handle.device_type == DeviceConfig_pb2.VISION
        ]
        assert len(vision_device_ids) == 1, "only 1 vision device is expected"
        vision_device_id = vision_device_ids[0]

        vision = VisionConfigClient(self.router)

        sensor_id = VisionConfig_pb2.SensorIdentifier()
        sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
        intrinsic = vision.GetIntrinsicParameters(sensor_id, vision_device_id)

        fx, fy = intrinsic.focal_length_x, intrinsic.focal_length_y
        cx, cy = intrinsic.principal_point_x, intrinsic.principal_point_y
        dc = intrinsic.distortion_coeffs
        dist = [dc.k1, dc.k2, dc.k3, dc.p1, dc.p2]

        K = [
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ]
        intrinsics = {"K": K, "distortion": dist}

        return intrinsics

    # try not to use this function because joint control has a different inverse kinematic solver than cartesian control
    def move_to_joint_angles(self, q: list, action_name: str) -> None:
        """
        Helper to send a reach_joint_angles action in joint space, wait for it, and clean up.
        Args:
            q (list): target joint angles (radians)
            action_name (str): name of the action (for logging)
        """

        action = Base_pb2.Action()
        action.name = action_name
        action.application_data = ""

        actuator_count = self.base.GetActuatorCount()

        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = q[joint_id]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        
        self.base.ExecuteAction(action)

        finished = e.wait(20)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Joints movement completed")
        else:
            print("Timeout on action notification wait")

    def reach_pose(self, robot_state, action_name: str) -> None:
        """
        Helper to send a reach_pose action in Cartesian coordinates (x,y,z,pitch,roll,yaw), wait for it, and clean up.
        
        Args:
            robot_state (KinovaState): target pose and gripper position
            action_name (str): name of the action (for logging)
        """
        # build action
        action = Base_pb2.Action()
        action.name = action_name
        action.application_data = ""
        # action.change_twist.angular = 35.0   # add 70°/s to whatever the current cap is

        # get current orientation for X/Y fallback
        pose = action.reach_pose.target_pose
        arm = robot_state
        # gripper = robot_state.gripper
        waypoint = Waypoint(data=[robot_state.data[0], robot_state.data[1], self.height])

        pose.x = float(arm[0])
        pose.y = float(arm[1])
        pose.z = float(arm[2])
        pose.theta_x = float(arm[3])
        pose.theta_y = float(arm[4])
        pose.theta_z = float(arm[5])

        # subscribe & execute
        e = threading.Event()
        handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        # self.go_to_waypoint(waypoint) # first go to above the target to avoid collisions
        self.base.ExecuteAction(action)

        finished = e.wait(20)
        if finished:
            print(f"[{action_name}] completed at {arm}")
            print("movement complete!")
        else:
            print(f"[{action_name}] timed out.")
        self.base.Unsubscribe(handle)

        return finished

    def pick_bowl(self, waypoint, z_angle: float = 0.01) -> None:
        """
        1. move down to pick height (6cm) at the given theta_z
        2. close gripper and wait until it actually grabs
        3. raise up to 25cm
        Args:
            waypoint (Waypoint): target (x,y) position to pick bowl from
            z_angle (float): end-effector yaw (theta_z) angle (degrees) to use when picking
        """
        init_cartesian_pose = self.get_current_state()

        # --- go down to pick ---
        pick_wp = np.array([waypoint.data[0], waypoint.data[1], 0.06,
        179.9, init_cartesian_pose.data[4], z_angle], dtype=np.float32)

        pick_state = KinovaState(data=pick_wp, gripper=0)
        finished = self.reach_pose(pick_state, "pick_bowl")

        # --- close gripper ---
        if not finished:
            print("Descent failed or timed out—aborting pick.")
            return

        gripper_cmd = Base_pb2.GripperCommand()
        gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = 1.0
        self.base.SendGripperCommand(gripper_cmd)

        start = time.time()
        while True:
            meas = self.base.GetMeasuredGripperMovement(self.gripper_request)
            if abs(meas.finger[0].value - finger.value) < 0.06:
                break
            if time.time() - start > 2:
                print("Warning: gripper closure timed out")
                break
            time.sleep(0.1)

        # check if it holds tightly
        meas = self.base.GetMeasuredGripperMovement(self.gripper_request)
        print("check tightly")
        if abs(meas.finger[0].value) < 0.93:
            finger.value = 0.95
            self.base.SendGripperCommand(gripper_cmd)

        self.pick_bowl_pose = self.get_current_state()

        # --- lift up ---
        finished = self.reach_pose(init_cartesian_pose, "raise_bowl")

        return finished

    def place_bowl(self) -> None:
        """
        1. move down to place height (6.4cm)
        2. open gripper and wait until it actually releases
        3. raise up to 25cm at the given theta_z
        """
        
        init_cartesian_pose = self.get_current_state()
        finished = self.reach_pose(self.pick_bowl_pose, "place_bowl")

        # --- open gripper ---
        if not finished:
            print("Descent failed or timed out—aborting place.")
            return

        gripper_cmd = Base_pb2.GripperCommand()
        gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = 0
        self.base.SendGripperCommand(gripper_cmd)

        start = time.time()
        while True:
            meas = self.base.GetMeasuredGripperMovement(self.gripper_request)
            if abs(meas.finger[0].value - finger.value) < 0.01:
                break
            if time.time() - start > 1.5:
                print("Warning: gripper open timed out")
                break
            time.sleep(0.1)

        # --- lift up ---
        self.reach_pose(init_cartesian_pose, "recover_from_place")

    def pick_spatula(self, waypoint, z_angle: float) -> None:
        """
        1. move down to pick height (3.5cm) at the given theta_z
        2. close gripper and wait until it actually grabs
        3. raise up to 25cm
        Args:
            waypoint (Waypoint): target (x,y) position to pick spatula from
            z_angle (float): end-effector yaw (theta_z) angle (degrees) to use when picking
        """
        # --- go down to pick ---
        init_cartesian_pose = self.get_current_state()

        pick_data = np.array([waypoint.data[0], waypoint.data[1], 0.035,
        init_cartesian_pose.data[3], init_cartesian_pose.data[4], z_angle], dtype=np.float32)
        pick_state = KinovaState(data=pick_data, gripper=0)

        finished = self.reach_pose(pick_state, "pick_spatula")

        # --- close gripper ---
        if not finished:
            print("Descent failed or timed out—aborting pick.")
            return
        
        gripper_cmd = Base_pb2.GripperCommand()
        gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = 1.0
        self.base.SendGripperCommand(gripper_cmd)

        start = time.time()
        while True:
            meas = self.base.GetMeasuredGripperMovement(self.gripper_request)
            if abs(meas.finger[0].value - finger.value) < 0.01:
                break
            if time.time() - start > 2:
                print("Warning: gripper closure timed out")
                break
            time.sleep(0.1)

        self.pick_spatula_pose = self.get_current_state()

        # --- lift up ---
        self.reach_pose(init_cartesian_pose, "raise_spatula")

    def place_spatula(self) -> None:
        """
        1. move down to place height (5cm)
        2. open gripper and wait until it actually releases
        3. raise up to 25cm at the given theta_z
        """
        # --- go down to place ---
        init_cartesian_pose = self.get_current_state()
        finished = self.reach_pose(self.pick_spatula_pose, "place_spatula")

        # --- open gripper ---
        if not finished:
            print("Descent failed or timed out—aborting place.")
            return

        gripper_cmd = Base_pb2.GripperCommand()
        gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = 0
        self.base.SendGripperCommand(gripper_cmd)

        start = time.time()
        while True:
            meas = self.base.GetMeasuredGripperMovement(self.gripper_request)
            if abs(meas.finger[0].value - finger.value) < 0.01:
                break
            if time.time() - start > 1.5:
                print("Warning: gripper open timed out")
                break
            time.sleep(0.1)

        # --- lift up ---
        self.reach_pose(init_cartesian_pose, "recover_spatula")

    def pour(self, waypoint) -> None:
        """
        1. move above pour location
        2. rotate end-effector to pour down
        3. rotate back to recover
        4. move back to initial position
        Args:
            waypoint (Waypoint): target (x,y) position to pour into
        """
        init_cartesian_pose = self.get_current_state()

        data: NDArray = np.array([waypoint.data[0], waypoint.data[1], 0.40, 102.4, -89, 79.6])
        gripper: float = init_cartesian_pose.gripper
        pose: KinovaState = KinovaState(data, gripper)

        self.reach_pose(pose, "pour_down")
        self.reach_pose(init_cartesian_pose, "recover_pour")

    def stir(self, waypoint) -> None:
        """ 
        Args:
            waypoint (Waypoint): target (x,y) position to stir at
        """
        init_cartesian_pose = self.get_current_state()
        twist_coordinates = KinovaState(data=np.array([waypoint.data[0], waypoint.data[1], 0.35, 
                                                       140, init_cartesian_pose.data[4], init_cartesian_pose.data[5]]), 
                                                       gripper=init_cartesian_pose.gripper)
        self.reach_pose(twist_coordinates,"going above stir location")

        #square shaped stirring
        y_correction = -0.04
        x_offset = 0.08
        y_offset = -0.08
        z_level = 0.17
        point1_wp = Waypoint(data=np.array([waypoint.data[0], waypoint.data[1] + y_correction, z_level]))
        point2_wp = Waypoint(data=np.array([waypoint.data[0] + x_offset, waypoint.data[1] + y_correction, z_level]))
        point3_wp = Waypoint(data=np.array([waypoint.data[0] + x_offset, waypoint.data[1] + y_correction + y_offset, z_level]))
        point4_wp = Waypoint(data=np.array([waypoint.data[0], waypoint.data[1] + y_correction + y_offset, z_level]))
        stirring_wp = [point1_wp, point2_wp, point3_wp, point4_wp]

        for i in range(3):
            for coordinate in stirring_wp:
                self.go_to_waypoint(coordinate)
                print(f"going to {coordinate}")

        self.reach_pose(init_cartesian_pose, "returning to pose before stirring")



from dataclasses import dataclass
import numpy as np
from typing_extensions import override
from numpy.typing import NDArray
import torch

from roboenv import Action


@dataclass
class KinovaAction(Action):
    """
    A dataclass representing actions executable by the kinova arm.
    Question: should this be velocity based or position based control?
    """

    data: NDArray  # [dx, dy, dz, dtx, dty, dtz]
    gripper: float  # desired gripper width, NOT change in width

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    @override
    def numpy(self) -> NDArray:
        return np.append(self.data, [self.gripper])

    @property
    @override
    def torch(self) -> torch.Tensor:
        return torch.from_numpy(self.numpy).to(self.device)


@dataclass
class Waypoint(Action):
    """
    A dataclass representing waypoints in the environment which the kinova arm can navigate to.
    """

    data: NDArray[np.float32]  # [px, py, pz] in robot coordinates

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        self.data[2] = max(0.25, self.data[2])

    @property
    @override
    def numpy(self) -> NDArray:
        return self.data

    @property
    @override
    def torch(self) -> torch.Tensor:
        raise ValueError("Why are you calling me?")

    def __repr__(self) -> str:
        return f"Waypoint({self.data[0]:.2f}, {self.data[1]:.2f}, {self.data[2]:.2f})"



from dataclasses import dataclass
from typing_extensions import override
from numpy.typing import NDArray
import torch
import numpy as np

from roboenv import State


@dataclass
class KinovaState(State):
    """
    A dataclass representing proprioceptive states of the kinova arm.
    """

    data: NDArray  # [px, py, pz, tx, ty, tz]
    gripper: float  # gripper width

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        """
        Set dtype to float32 to save some memory.
        """
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)

    @property
    @override
    def numpy(self) -> NDArray:
        return np.append(self.data, [self.gripper]).astype(np.float32)

    @property
    @override
    def torch(self) -> torch.Tensor:
        return torch.from_numpy(self.numpy).to(device=self.device, dtype=torch.float32)

