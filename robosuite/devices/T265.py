"""
Driver class for T265 controller.
"""

import threading
import time

import numpy as np
import pyrealsense2.pyrealsense2 as rs
from pynput.keyboard import Controller, Key, Listener
from scipy.spatial.transform import Rotation as RRR

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class KeyboardForT265:
    def __init__(self):

        self._reset_state = 0
        self.grasp = False

        # launch a new listener thread to listen to keyboard
        self.thread = Listener(on_press=self._on_press, on_release=self._on_release)
        self.thread.start()

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset grasp
        self.grasp = False

    def _on_press(self, key):
        pass

    def _on_release(self, key):
        try:
            # controls for grasping
            if key.char == "p":
                self.grasp = not self.grasp  # toggle gripper

            # user-commanded reset
            elif key.char == "0":
                self._reset_state = 1
                self._reset_internal_state()

        except AttributeError as e:
            pass


class T265(Device):
    """
    A minimalistic driver class for a T265.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0):

        self.pipeline = rs.pipeline()
        config = rs.config()

        context = rs.context()
        devices = context.query_devices()
        print(devices.size(), devices[0].get_info(rs.camera_info.name))  # 1 Intel RealSense T265

        config.enable_stream(rs.stream.pose)
        self.pipeline.start(config)

        self.init_pos = False

        self._display_controls()

        self._reset_state = 0
        self._enabled = False

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        self.keyboard = KeyboardForT265()
        self.grasp = False

        # launch a new listener thread to listen to T265
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Your hand", "Hold its handle and move")
        print_command("keyboard 0", "Reset")
        print_command("keyboard p", "Grasp")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """

        dpos = (self.pos - self.last_pos) * self.pos_sensitivity
        self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        ) * self.rot_sensitivity  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
        )

    def run(self):
        """Listener method that keeps pulling new messages."""
        while True:
            frames = self.pipeline.wait_for_frames()
            pose = frames.get_pose_frame()
            if pose and self._enabled:
                data = pose.get_pose_data()

                # transformation matrix from T265 to robot base
                mat_trans = np.eye(4)
                mat_trans[:3, :3] = np.copy((RRR.from_quat([0.5, -0.5, -0.5, 0.5])).as_matrix())  # T265

                # current T265 pose
                mat_se3 = np.eye(4)
                mat_se3[:3, :3] = np.copy(
                    (RRR.from_quat([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])).as_matrix()
                )
                mat_se3[:3, 3] = np.array([data.translation.x, data.translation.y, data.translation.z])

                # initial T265 pose
                if not self.init_pos:
                    mat_se3_base = np.eye(4)
                    mat_se3_base = np.linalg.inv(mat_trans @ mat_se3) @ mat_se3_base
                    self.init_pos = True

                # calculate T265 pose in robot base
                trk_mat_se3 = mat_trans @ mat_se3 @ mat_se3_base

                self.pos = trk_mat_se3[:3, 3]
                self.quat = np.copy(RRR.from_matrix(trk_mat_se3[:3, :3]).as_quat())
                self.raw_drotation = np.copy(RRR.from_quat(self.quat).as_euler("xyz"))
                self.rotation = np.copy(trk_mat_se3[:3, :3])

                # check grasp (via keyboard)
                self.grasp = np.copy(self.keyboard.grasp)

                # check reset (via keyboard)
                self._reset_state = self.keyboard._reset_state
