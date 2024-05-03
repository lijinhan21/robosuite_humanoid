import argparse
import json
import math
import os
import pickle
import time
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import tabulate
import yaml
from mujoco import viewer
from retarget.utils.configs import load_config

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="HumanoidSimple")
    args = parser.parse_args()
    return args


def gripper_joint_pos_controller(obs, desired_qpos, kp=100, damping_ratio=1):
    """
    Calculate the torques for the joints position controller.

    Args:
        obs: dict, the observation from the environment
        desired_qpos: np.array of shape (12, ) that describes the desired qpos (angles) of the joints on hands, right hand first, then left hand

    Returns:
        desired_torque: np.array of shape (12, ) that describes the desired torques for the joints on hands
    """
    # get the current joint position and velocity
    actuator_idxs = [0, 1, 4, 6, 8, 10]
    # order:
    # 'gripper0_right_joint_r_thumb_proximal_1', 'gripper0_right_joint_r_thumb_proximal_2', 'gripper0_right_joint_r_thumb_middle', 'gripper0_right_joint_r_thumb_distal', 'gripper0_right_joint_r_index_proximal', 'gripper0_right_joint_r_index_distal', 'gripper0_right_joint_r_middle_proximal', 'gripper0_right_joint_r_middle_distal', 'gripper0_right_joint_r_ring_proximal', 'gripper0_right_joint_r_ring_distal', 'gripper0_right_joint_r_pinky_proximal', 'gripper0_right_joint_r_pinky_distal''gripper0_right_joint_r_thumb_proximal_1', 'gripper0_right_joint_r_thumb_proximal_2', 'gripper0_right_joint_r_thumb_middle', 'gripper0_right_joint_r_thumb_distal', 'gripper0_right_joint_r_index_proximal', 'gripper0_right_joint_r_index_distal', 'gripper0_right_joint_r_middle_proximal', 'gripper0_right_joint_r_middle_distal', 'gripper0_right_joint_r_ring_proximal', 'gripper0_right_joint_r_ring_distal', 'gripper0_right_joint_r_pinky_proximal', 'gripper0_right_joint_r_pinky_distal'
    joint_qpos = np.concatenate(
        (obs["robot0_right_gripper_qpos"][actuator_idxs], obs["robot0_left_gripper_qpos"][actuator_idxs])
    )
    joint_qvel = np.concatenate(
        (obs["robot0_right_gripper_qvel"][actuator_idxs], obs["robot0_left_gripper_qvel"][actuator_idxs])
    )

    position_error = desired_qpos - joint_qpos
    vel_pos_error = -joint_qvel

    # calculate the torques: kp * position_error + kd * vel_pos_error
    kd = 2 * np.sqrt(kp) * damping_ratio - 10
    desired_torque = np.multiply(np.array(position_error), np.array(kp)) + np.multiply(vel_pos_error, kd)

    # clip and rescale to [-1, 1]
    desired_torque = np.clip(desired_torque, -1, 1)

    return desired_torque


class HandPose:
    def __init__(self):
        self.thumb0 = None
        self.thumb = None
        self.index = None
        self.middle = None
        self.ring = None
        self.pinky = None

    def set_pose(self, thumb0, thumb, index, middle, ring, pinky):
        self.thumb0 = thumb0
        self.thumb = thumb
        self.index = index
        self.middle = middle
        self.ring = ring
        self.pinky = pinky


class GraspPrimitive:
    def __init__(self, config):
        self.name_to_pose = {}

        self.thumb0_config = config.thumb0_config
        self.thumb_config = config.thumb_config
        self.finger_config = config.finger_config

        for key, value in self.thumb0_config.items():
            self.thumb0_config[key] = value / 180 * np.pi
        for key, value in self.thumb_config.items():
            self.thumb_config[key] = value / 180 * np.pi
        for key, value in self.finger_config.items():
            self.finger_config[key] = value / 180 * np.pi

        print("thumb0_config", self.thumb0_config)

        self.add_from_dict(config.grasp_primitives)

        # self.thumb0_config = {"in": thumb0_range[1], "out": thumb0_range[0]}
        # self.thumb_config = {
        #     "extend": thumb_range[0],
        #     "slightly_bent": thumb_range[1],
        #     "bent": thumb_range[2],
        #     "folded": thumb_range[3],
        #     "tucked": thumb_range[4],
        # }
        # self.finger_config = {
        #     "extend": finger_range[0],
        #     "slightly_bent": finger_range[1],
        #     "bent": finger_range[2],
        #     "folded": finger_range[3],
        #     "tucked": finger_range[4],
        # }

    def add_primitive(self, name, pose):
        self.name_to_pose[name] = pose

    def add_from_dict(self, dict):
        for name, pose in dict.items():
            hand_pose = HandPose()
            hand_pose.set_pose(
                pose["thumb0"], pose["thumb"], pose["index"], pose["middle"], pose["ring"], pose["pinky"]
            )
            self.add_primitive(name, hand_pose)

    def save_as_dict(self):
        dict = {}
        for name, pose in self.name_to_pose.items():
            dict[name] = {
                "thumb0": pose.thumb0,
                "thumb": pose.thumb,
                "index": pose.index,
                "middle": pose.middle,
                "ring": pose.ring,
                "pinky": pose.pinky,
            }

        ret = {"thumb0_config": {}, "thumb_config": {}, "finger_config": {}, "grasp_primitives": dict}
        for key, value in self.thumb0_config.items():
            print(key, value)
            ret["thumb0_config"][key] = int(value * 180 / np.pi)
        for key, value in self.thumb_config.items():
            ret["thumb_config"][key] = int(value * 180 / np.pi)
        for key, value in self.finger_config.items():
            ret["finger_config"][key] = int(value * 180 / np.pi)

        return ret

    def get_joint_angles(self, name):
        pose = self.name_to_pose[name]
        thumb0 = self.thumb0_config[pose.thumb0]
        thumb = self.thumb_config[pose.thumb]
        index = self.finger_config[pose.index]
        middle = self.finger_config[pose.middle]
        ring = self.finger_config[pose.ring]
        pinky = self.finger_config[pose.pinky]

        return np.array([thumb0, thumb, index, middle, ring, pinky])

    def real_control_list(self):
        def map_to_real_control(value):
            # original: 0 - np.pi/2, real: 1000 - 0
            return 1000 - 1000 * value / (np.pi / 2)

        ret = {}
        for name, pose in self.name_to_pose.items():
            thumb0 = map_to_real_control(self.thumb0_config[pose.thumb0])
            thumb = map_to_real_control(self.thumb_config[pose.thumb])
            index = map_to_real_control(self.finger_config[pose.index])
            middle = map_to_real_control(self.finger_config[pose.middle])
            ring = map_to_real_control(self.finger_config[pose.ring])
            pinky = map_to_real_control(self.finger_config[pose.pinky])
            ret[name] = [thumb0, thumb, index, middle, ring, pinky]

        return ret


def main():
    args = get_args()

    config = {}
    config["robots"] = "GR1UpperBody"
    config["env_name"] = args.environment
    config["env_configuration"] = "bimanual"
    config["controller_configs"] = load_controller_config(default_controller="JOINT_POSITION")
    config["controller_configs"]["kp"] = 500

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=["frontview", "agentview", "handview"],
    )
    obs = env.reset()

    mjmodel = env.sim.model._model
    mjdata = env.sim.data._data

    # load grasp primitives
    grasp_config = load_config("scripts/grasp_primitives.yaml")
    grasp_dict = GraspPrimitive(grasp_config)

    real_control_dict = grasp_dict.real_control_list()
    with open("scripts/grasp_primitives_real.yaml", "w") as f:
        yaml_str = yaml.dump(real_control_dict, default_flow_style=False, sort_keys=False)
        f.write(yaml_str)

    # # 1. Large Diameter (Cylindrical Grasp)
    # hand_pose = HandPose()
    # hand_pose.set_pose("in", "slightly_bent", "slightly_bent", "slightly_bent", "slightly_bent", "slightly_bent")
    # grasp_dict.add_primitive("large_diameter", hand_pose)

    # # 2. Small Diameter (Cylindrical Grasp)
    # hand_pose = HandPose()
    # hand_pose.set_pose("in", "slightly_bent", "folded", "folded", "folded", "folded")
    # grasp_dict.add_primitive("small_diameter", hand_pose)

    # # 9. Palmar Pinch
    # hand_pose = HandPose()
    # hand_pose.set_pose("in", "slightly_bent", "bent", "extend", "extend", "extend")
    # grasp_dict.add_primitive("palmar_pinch", hand_pose)

    # # 8. Prismatic Finger
    # hand_pose = HandPose()
    # hand_pose.set_pose("in", "slightly_bent", "bent", "bent", "tucked", "tucked")
    # grasp_dict.add_primitive("prismatic_finger", hand_pose)

    # # 27. Quadpod
    # hand_pose = HandPose()
    # hand_pose.set_pose("in", "slightly_bent", "slightly_bent", "slightly_bent", "slightly_bent", "extend")
    # grasp_dict.add_primitive("quadpod", hand_pose)

    # # 5. Light Tool
    # hand_pose = HandPose()
    # hand_pose.set_pose("out", "slightly_bent", "tucked", "tucked", "tucked", "tucked")
    # grasp_dict.add_primitive("light_tool", hand_pose)

    # # 12. Precision Disk
    # hand_pose = HandPose()
    # hand_pose.set_pose("out", "slightly_bent", "slightly_bent", "slightly_bent", "slightly_bent", "slightly_bent")
    # grasp_dict.add_primitive("precision_disk", hand_pose)

    # # 18. Extension Type
    # hand_pose = HandPose()
    # hand_pose.set_pose("in", "slightly_bent", "tucked", "tucked", "tucked", "tucked")
    # grasp_dict.add_primitive("extension_type", hand_pose)

    # save dict
    with open("scripts/grasp_primitives_saved.yaml", "w") as f:
        yaml_str = yaml.dump(grasp_dict.save_as_dict(), default_flow_style=False, sort_keys=False)
        f.write(yaml_str)

    # # Function to overlay an image on top-left corner of the video frame
    # def overlay_image(frame, image):
    #     # Resize image to fit the frame
    #     max_height = 250  # Adjust size based on your need
    #     scale_factor = max_height / image.shape[0]
    #     resized_image = cv2.resize(image, (int(image.shape[1] * scale_factor), max_height))

    #     # Get dimensions
    #     ih, iw, _ = resized_image.shape
    #     fh, fw, _ = frame.shape

    #     # Ensure the image does not exceed the frame size
    #     if ih <= fh and iw <= fw:
    #         # Overlay the image
    #         frame[:ih, :iw] = cv2.addWeighted(frame[:ih, :iw], 0.05, resized_image, 0.95, 0)
    #     return frame

    # names = ["large_diameter", "small_diameter", "palmar_pinch", "prismatic_finger", "quadpod", "light_tool", "precision_disk", "extension_type"]
    # # Prepare video outputs
    # output_videos = []

    # # Process each video and image pair
    # for name in names:
    #     cap = cv2.VideoCapture(f"grasp_video/{name}.mp4")
    #     image = cv2.imread(f"grasp_video/{name}")
    #     if image is None or not cap.isOpened():
    #         continue

    #     # Define the codec and create VideoWriter object
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     out = cv2.VideoWriter(f"grasp_video/{name}_overlay.mp4", fourcc, 30.0, (640, 640))

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # Overlay image on frame
    #         frame = overlay_image(frame, image)
    #         out.write(frame)

    #     # Release everything
    #     cap.release()
    #     out.release()
    #     output_videos.append(f"grasp_video/{name}_overlay.mp4")

    # # Assuming all videos are of the same dimension and frame rate
    # # Combine videos in a grid
    # final_frames = []
    # cap_list = [cv2.VideoCapture(vid) for vid in output_videos]

    # # Check if all captures are opened
    # if all(cap.isOpened() for cap in cap_list):
    #     while True:
    #         frames = []
    #         for cap in cap_list:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 frames.append(None)
    #             else:
    #                 frames.append(frame)

    #         if any(f is None for f in frames):
    #             break

    #         # Arrange frames in a 4x2 grid
    #         top_row = np.hstack(frames[:4])
    #         bottom_row = np.hstack(frames[4:])
    #         final_frame = np.vstack((top_row, bottom_row))
    #         print("final_frame", final_frame.shape)
    #         final_frames.append(final_frame)

    # # Assuming the same frame rate and size for the output
    # out_final = cv2.VideoWriter('grasp_video/final_output.mp4', fourcc, 30.0, final_frames[0].shape[1::-1])

    # for frame in final_frames:
    #     out_final.write(frame)
    # print("len of final_frames", len(final_frames))

    # # Release final output and all captures
    # out_final.release()
    # for cap in cap_list:
    #     cap.release()

    # print("Video stitching completed!")

    # exit(0)

    grasp_type = "precision_disk"
    os.makedirs("grasp_video", exist_ok=True)
    writer = cv2.VideoWriter(f"grasp_video/{grasp_type}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 640))

    fps = 20
    thumb_wait_time = 50
    idx = 0

    with mujoco.viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:

        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
            viewer.cam.azimuth = 180
            viewer.cam.lookat = np.array([0.0, 0.0, 1.5])
            viewer.cam.distance = 1.0
            viewer.cam.elevation = -35

        while viewer.is_running():

            action = np.zeros(32)

            joint_pos = np.concatenate((grasp_dict.get_joint_angles(grasp_type), np.zeros(6)))
            print("joint_pos", joint_pos)
            action[10:16] = gripper_joint_pos_controller(obs, joint_pos)[:6]

            idx += 1
            if idx < thumb_wait_time:
                action[11] = 0

            obs, reward, done, _ = env.step(action)

            # save image
            img = obs["handview_image"]
            img = cv2.flip(img, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            viewer.sync()
            cv2.imshow("image", img)
            cv2.waitKey(10)

            img = cv2.resize(img, (640, 640))
            writer.write(img)

            time.sleep(1 / fps)
            if idx > 100:
                break

        writer.release()


if __name__ == "__main__":
    main()
