"""
kinova.py

An implementation of the Robot class with the Kinova robotic arm.
hi matthew and ali please work on this
hi matthew
"""

# run from ~/Projects/emg-gaze/
# uv run python -m src.environment.kinova

# home coords xyz below approx. (0.5, 0, 0.43)
# KinovaState(data=array([ 5.7614034e-01, -1.5542102e-02,  4.2999908e-01,  9.0551262e+01, 1.0861688e+00,  8.7556015e+01], dtype=float32), gripper=0.5, device=device(type='cpu'))

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override
import sys
import time
import threading
from roboenv import Robot


from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, VisionConfig_pb2, DeviceConfig_pb2
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
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
        self.id: int = id
        self.ip: str = "192.168.1.10"
        self.state = KinovaState([0.6, 0, 0.45, 179, 0, 0], 0.5)
        self.bounds = [(0.05, 1.00), (-0.60, 0.65), (0.02, 0.60)]
        


        self.connected = False
        self.busy = False
        # self.nb_dof = 0
        self.action = Base_pb2.Action()

        # self.gripper_command = Base_pb2.GripperCommand()
        # self.gripper_command.mode = Base_pb2.GRIPPER_POSITION
        self.gripper_request = Base_pb2.GripperRequest()
        self.gripper_request.mode = Base_pb2.GRIPPER_POSITION

        self.notification_handles = []
        self.e = None

        self.error_callback = lambda kException: print("_________ callback error _________ {}".format(kException))
        self.tcp_client = None
        self.router_client = None
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
        UDP_PORT = 10001

        if self.connected:
            self.shutdown()

        try:
            self.tcp_client = TCPTransport()
            self.router_client = RouterClient(self.tcp_client, self.error_callback)
            # self.router_client = RouterClient(self.transport, RouterClient.basicErrorCallback)
            self.tcp_client.connect(self.ip, TCP_PORT)

            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = "admin"
            session_info.password = "admin"
            session_info.session_inactivity_timeout = 60000  # milliseconds
            session_info.connection_inactivity_timeout = 2000  # milliseconds

            self.session_manager = SessionManager(self.router_client)
            self.session_manager.CreateSession(session_info)

            print("Session created")

            self.base = BaseClient(self.router_client)
            self.base_cyclic = BaseCyclicClient(self.router_client)
            # self.device_config_client = DeviceConfigClient(self.router_client)
            # self.control_config_client = BaseClient(self.router_client)

            # self.nb_dof = self.base.GetActuatorCount().count

            servoing_mode = Base_pb2.ServoingModeInformation()
            servoing_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(servoing_mode)

            self.connected = True

            # print(f"hello I am robot id {self.id}")
            # return True

        except KException as ex:
            self.on_error(ex)
            # return False

    @override
    def shutdown(self) -> None:
        """
        Safely disconnect robot.

        Terminates active session and client objects.
        """
        
        # if self.session_manager != None:
        if self.connected:
            self.session_manager.CloseSession()
            self.router_client.SetActivationStatus(False)
            self.tcp_client.disconnect()
            # # Implementation in example code:
            # router_options = RouterClientSendOptions()
            # router_options.timeout_ms = 1000
            # self.session_manager.CloseSession()
            # self.tcp_client.disconnect()

        self.connected = False
        self.base = None
        # self.device_config_client = None
        # self.control_config_client = None
        self.session_manager = None
        self.router_client = None
        self.tcp_client = None

        # print(f"bye bye from robot {self.id}")

    @override
    def get_current_state(self, joints=False) -> KinovaState:
        """
        Query the robot's current state.

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
        
        return KinovaState(proprio, gripper)

    @override
    def execute_action(self, action: KinovaAction) -> None:
        """
        Execute a control action on the robot.

        Twist command specifies velocities; m/s and deg/s for our purposes.

        Args:
            action (Action): The action (twist command & gripper position) to send.
        """

        # print(action.gripper)
        if action.gripper > 1.0 or action.gripper < 0.0:
            print("invalid gripper command")
            return
        
        
        command = Base_pb2.TwistCommand()

        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = 0  # duration constraint
        twist = command.twist

        # bound the movement in allowed area
        if not self.in_bounds():
            print("Predicted position out of bounds, stopping action")
            self.base.Stop()
            twist.linear_x = 0
            twist.linear_y = 0
            twist.linear_z = 0
            twist.angular_x = 0
            twist.angular_y = 0
            twist.angular_z = 0
            self.base.SendTwistCommand(command)
            sys.exit(1)
        else:
            twist.linear_x = action.data[0]  # should be m/s
            twist.linear_y = action.data[1]
            twist.linear_z = action.data[2]
            twist.angular_x = action.data[3]  # should be deg/s
            twist.angular_y = action.data[4]
            twist.angular_z = action.data[5]

        self.base.SendTwistCommand(command)

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
        # print("Gripper going to position {:0.2f}...".format(finger.value))
        self.base.SendGripperCommand(gripper_command)
        # time.sleep(1)

        # print(f"action executed! I am robot {self.id}")

    @override
    def go_to_waypoint(self, waypoint: Waypoint) -> None:
        """
        Command the robot to move to a specific Cartesian waypoint.
        
        Does not change end-effector orientation (theta_x, theta_y, theta_z).

        Args:
            waypoint (Any): The target waypoint (x, y, z).
        """
        # print(f"going to waypoint! {self.id}")
        self.action.Clear()
        self.action = Base_pb2.Action()
        self.action.name = "go_to_waypoint"
        self.action.application_data = ""

        feedback = self.base_cyclic.RefreshFeedback()
        pose = self.action.reach_pose.target_pose

        # check if the robot is in bounds, else ends the program
        if not self.in_bounds(waypoint):
            pose.x = feedback.base.tool_pose_x
            pose.y = feedback.base.tool_pose_y
            pose.z = feedback.base.tool_pose_z
            pose.theta_x = feedback.base.tool_pose_theta_x
            pose.theta_y = feedback.base.tool_pose_theta_y
            pose.theta_z = feedback.base.tool_pose_theta_z
            print("Waypoint is out of bounds, keeping current position")
        else:
            pose.x = float(waypoint[0])
            pose.y = float(waypoint[1])
            pose.z = float(waypoint[2])
            pose.theta_x = feedback.base.tool_pose_theta_x
            pose.theta_y = feedback.base.tool_pose_theta_y
            pose.theta_z = feedback.base.tool_pose_theta_z
        # OPTIONAL: orientation (pose. theta_x, theta_y, theta_z)
        # NB not sure if pose.theta_xyz need to have values

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.notification_handles.append(notification_handle)

        self.base.ExecuteAction(self.action)

        finished = e.wait(20)  # wait for up to 20 seconds or until action is completed
        self.base.Unsubscribe(notification_handle)

        # if finished:
        #     print(f"arrived at waypoint! {self.id}")

    # Note: Waypoint is defined in ../agent/aria.py
    # first in_bounds is for velocity controlled movement (e.g. TwistCommand())
    def in_bounds(self, waypoint: Waypoint = None) -> bool:
        """
        Returns True if the robot is within the operation area.
        If `waypoint` is provided, check that instead of the current state.
        """
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        if waypoint is not None:
            x, y, z = waypoint.data
        else:
            x, y, z = self.get_current_state().data[:3]
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
            print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()

        return check

    def execute_seq(self):
    
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

        if finished:
            print("Home position reached")
        else:
            print("Could not reach position in time (10 seconds right now)")

    def go_home(self):
        """
        Go to designated home waypoint.
        """

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.notification_handles.append(notification_handle)

        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "HOME":
                action_handle = action.handle
        
        self.base.ExecuteActionFromReference(action_handle)

        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = 0.0
        # print("Gripper going to position {:0.2f}...".format(finger.value))
        self.base.SendGripperCommand(gripper_command)

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
        device_manager = DeviceManagerClient(self.router_client)
        device_handles = device_manager.ReadAllDevices()
        vision_device_ids = [
            handle.device_identifier for handle in device_handles.device_handle
            if handle.device_type == DeviceConfig_pb2.VISION
        ]
        assert len(vision_device_ids) == 1, "only 1 vision device is expected"
        vision_device_id = vision_device_ids[0]

        vision = VisionConfigClient(self.router_client)

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


    def move_to_joint_angles(self, action_name, q):
        
        action = Base_pb2.Action()
        action.name = action_name
        action.application_data = ""

        actuator_count = self.base.GetActuatorCount()

        # Place arm straight up
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
        return finished


    def _execute_reach(self, waypoint, action_name, theta_x=None, theta_y=None, theta_z=None):
        """
        Helper to send a reach_pose action, wait for it, and clean up.
        """
        # build action
        action = Base_pb2.Action()
        action.name = action_name
        action.application_data = ""
        # action.change_twist.angular = 35.0   # add 70°/s to whatever the current cap is

        # get current orientation for X/Y fallback
        fb = self.base_cyclic.RefreshFeedback()
        pose = action.reach_pose.target_pose

        # TODO: @Matthew to add the adjusted bound
        pose.x = float(waypoint[0])
        pose.y = float(waypoint[1])
        pose.z = float(waypoint[2])
        pose.theta_x = theta_x if theta_x is not None else fb.base.tool_pose_theta_x
        pose.theta_y = theta_y if theta_y is not None else fb.base.tool_pose_theta_y
        pose.theta_z = theta_z if theta_z is not None else fb.base.tool_pose_theta_z

        # subscribe & execute
        done = threading.Event()
        handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(done),
            Base_pb2.NotificationOptions()
        )
        self.base.ExecuteAction(action)

        finished = done.wait(20)
        if finished:
            print(f"[{action_name}] completed at {waypoint} with θx={theta_x}°, θy={theta_y}°, θz={theta_z}°")
        else:
            print(f"[{action_name}] timed out.")
        self.base.Unsubscribe(handle)
        
        return finished


    def pick_bowl(self, target_xyz, z_angle=0.01):
        """
        1. move down to pick height (6cm) at the given theta_z
        2. close gripper and wait until it actually grabs
        3. raise up to 25cm
        """
        self.pick_bowl_joint_angles = self.get_current_state(joints=True)

        # --- go down to pick ---
        pick_wp = [target_xyz[0], target_xyz[1], 0.06]
        finished = self._execute_reach(pick_wp, "pick_bowl", theta_z=z_angle)

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
            print("here")

        # --- lift up ---
        lift_wp = [target_xyz[0], target_xyz[1], 0.25]
        finished = self._execute_reach(lift_wp, "raise_bowl")

        return finished

    def place_bowl(self, target_xyz):
        """
        1. move down to place height (6.4cm)
        2. open gripper and wait until it actually releases
        3. raise up to 25cm at the given theta_z
        """
        
        # --- go down to place ---
        place_wp = [target_xyz[0], target_xyz[1], 0.20]
        finished = self._execute_reach(place_wp, "place_bowl0")

        place_wp = [target_xyz[0], target_xyz[1], 0.064]
        finished = self._execute_reach(place_wp, "place_bowl1")

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
        # lift_wp = [target_xyz[0], target_xyz[1], 0.25]
        
        # self._execute_reach(lift_wp, "recover_bowl", theta_z=90)

        # init_joint_pose = self.get_current_state(joints=True)
        # init_joint_pose[6] = 92.94
        self.move_to_joint_angles("recover_bowl", self.pick_bowl_joint_angles)


    def pick_spatula(self, target_xyz, z_angle):
        """
        1. move down to pick height (3.5cm) at the given theta_z
        2. close gripper and wait until it actually grabs
        3. raise up to 25cm
        """
        # --- go down to pick ---
        pick_wp = [target_xyz[0], target_xyz[1], 0.035]
        finished = self._execute_reach(pick_wp, "pick_spatula", theta_z=z_angle)

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

        # --- lift up ---
        lift_wp = [target_xyz[0], target_xyz[1], 0.25]
        self._execute_reach(lift_wp, "raise_spatula")


    def place_spatula(self, target_xyz):
        """
        1. move down to place height (5cm)
        2. open gripper and wait until it actually releases
        3. raise up to 25cm at the given theta_z
        """
        # --- go down to place ---
        place_wp = [target_xyz[0], target_xyz[1], 0.20]
        finished = self._execute_reach(place_wp, "place_spatula0")
        place_wp = [target_xyz[0], target_xyz[1], 0.05]
        finished = self._execute_reach(place_wp, "place_spatula1")

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
        lift_wp = [target_xyz[0], target_xyz[1], 0.25]
        self._execute_reach(lift_wp, "recover_spatula", theta_z=88.3)


    def pour(self, target_xyz):
        init_joint_pose = self.get_current_state(joints=True)

        pour_coord = [target_xyz[0], target_xyz[1], 0.40]

        self._execute_reach(pour_coord, "pour_down", theta_x=102.4, theta_y=-89, theta_z=79.6)
        self.move_to_joint_angles("recover_pour", init_joint_pose)


    def stir(self, target_xyz):
        init_pose = self.get_current_state().data
        init_coords = init_pose[:3]
        init_theta_x = init_pose[3]
        init_theta_y = init_pose[4]
        init_theta_z = init_pose[5]
        
        twist_coordinates = [target_xyz[0], target_xyz[1], 0.35]
        self._execute_reach(twist_coordinates,"adjusting pose", theta_x=140)
        #square shaped stirring
        y_correction = -0.04
        x_offset = 0.08
        y_offset = -0.08
        z_level = 0.17
        point1_coordinates = [target_xyz[0], target_xyz[1] + y_correction, z_level]
        point2_coordinates = [target_xyz[0] + x_offset, target_xyz[1] + y_correction, z_level]
        point3_coordinates = [target_xyz[0] + x_offset, target_xyz[1] + y_correction + y_offset, z_level]
        point4_coordinates = [target_xyz[0], target_xyz[1] + y_correction + y_offset, z_level]

        for i in range(3):
            for coordinate in [point1_coordinates, point2_coordinates, point3_coordinates, point4_coordinates, point1_coordinates]:
                self._execute_reach(coordinate, f"going to point {coordinate}")
        self._execute_reach([init_coords[0], init_coords[1], 0.30], "moving up")
        self._execute_reach(init_coords, "returning to initial pose", theta_x=179, theta_y=init_theta_y, theta_z=init_theta_z)


if __name__ == "__main__":
    robot = Kinova(10, 1)
    robot.launch()
    if robot.connected:
        print("Connection successful")
    print(robot.get_current_state())

    # wp1 = Waypoint([0.7, 0.3, 0.46])
    # robot.go_to_waypoint(wp1)
    # wp2 = Waypoint([0.6, 0, 0.45])
    #robot.go_to_waypoint(wp2)
    # a1 = KinovaAction(data=[-0.1, -0.3, -0.01, 0, 0, 0], gripper=0.7)
    # robot.execute_action(a1)
    # # a2 = KinovaAction(data=[0.1, 0.3, 0.01, 0, 0, 0], gripper=1.0)
    # # robot.execute_action(a2)
    print(robot.get_current_state())  # [0.7, 0.3, 0.46,0,0,0] 1
    robot.shutdown()


from dataclasses import dataclass
import numpy as np
from typing_extensions import override
from numpy.typing import NDArray
import torch

from roboenv import Action
from .myo_calibrate import PADDLE_POSES, FIST_POSES


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
class AriaSignal(Action):
    """
    A dataclass representing gaze signal from the Aria glasses.
    """

    # 2D or 3D gaze signal, up to Xu
    gaze_signal: NDArray[np.float32]

    @property
    @override
    def numpy(self) -> NDArray:
        return self.gaze_signal

    @property
    @override
    def torch(self) -> torch.Tensor:
        raise ValueError("Why are you calling me?")


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


@dataclass
class MyoSignal(Action):
    """
    A dataclass representing signals from the Myoband.
    Question: should this be raw signals or SVM-classfied classes?
        -> if raw signal, then should SVM classification be handled by Myoband class?
        -> or should the SVM classification by handled completely outside of the classes in this file?
    """

    paddle: int
    fist: int
    warn: str = ""

    def __post_init__(self) -> None:
        self.paddle = int(self.paddle)
        self.fist = int(self.fist)

    @property
    @override
    def numpy(self) -> NDArray:
        return np.array([self.paddle, self.fist])

    @property
    @override
    def torch(self) -> torch.Tensor:
        return torch.tensor([self.paddle, self.fist])

    def __repr__(self) -> str:
        if not self:
            paddle_pos = "NONE"
            fist_pos = "NONE"
        else:
            paddle_pos = PADDLE_POSES[self.paddle]
            fist_pos = FIST_POSES[self.fist]

        ret = f"\rPaddle: {paddle_pos:<7} | Fist:     {fist_pos:<7}"
        if len(self.warn) > 0:
            ret += f" | {self.warn:<100}"
        return ret

    def __bool__(self):
        return self.paddle != -1 and self.fist != -1  # if value is -1 something is wrong



from dataclasses import dataclass
from PIL import Image
from typing_extensions import override
from numpy.typing import NDArray
import torch
from torchvision.transforms import ToTensor
import numpy as np
from .data.image_utils import resnet18_transforms

from roboenv import State
from pyrealsense2.pyrealsense2 import composite_frame


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


@dataclass
class KinovaImage(State):
    """
    A dataclass representing images from the kinova arm camera.
    """

    data: NDArray[np.uint8]  # [720, 1280, 3], integers in range [0, 255]

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def pil(self) -> Image.Image:
        return Image.fromarray(self.data)

    @property
    @override
    def numpy(self) -> NDArray:
        return resnet18_transforms(self.pil).numpy()

    @property
    @override
    def torch(self) -> torch.Tensor:
        """
        Return images as [3, 224, 224], float32 in range [0, 1],
        as processed by Resnet18.
        """
        return resnet18_transforms(self.pil).to(self.device)


@dataclass
class RealSenseImage(State):
    """
    A dataclass representing camera states of the realsense cameras.
    """

    # put things here, probably some images and some metadata (e.g. timestamps).
    image: NDArray
    frame: composite_frame

    @property
    def pil(self) -> Image.Image:
        return Image.fromarray(self.image)
    
    @property
    @override
    def numpy(self) -> NDArray:
        return self.image

    @property
    def pp_numpy(self) -> NDArray:
        return resnet18_transforms(self.pil).numpy()
    
    @property
    @override
    def torch(self) -> torch.Tensor:
        return torch.from_numpy(self.image)

    @property
    def pp_torch(self):
        return resnet18_transforms(self.pil).to(self.device)
    
@dataclass
class RealSenseState(State):
    """
    A dataclass representing camera states of the realsense cameras.
    """

    # put things here, probably some images and some metadata (e.g. timestamps).
    _color_image: NDArray
    _depth_image: NDArray
    _color_frame: composite_frame
    _depth_frame: composite_frame

    def __post_init__(self):
        self.color = RealSenseImage(self._color_image, self._color_frame)
        self.depth = RealSenseImage(self._depth_image, self._depth_frame)

    @property
    @override
    def numpy(self) -> NDArray:
        pass

    @property
    @override
    def torch(self) -> torch.Tensor:
        pass


@dataclass
class Observation(State):
    """
    This is the highest-level wrapper for states. This is what HotPotEnv.step() returns.
    """

    state: KinovaState
    eye_image: KinovaImage
    cam_image: RealSenseImage

    @property
    @override
    def numpy(self) -> NDArray:
        return np.array([])

    @property
    @override
    def torch(self) -> torch.Tensor:
        return torch.tensor([])