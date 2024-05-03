import argparse
import json
import os
import pickle
import time
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import tabulate
from mujoco import viewer
from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config
from retarget.utils.constants import name_to_urdf_idx

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="HumanoidSimple")
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        help="Choice of interpolate algorithm. Can be 'linear' or 'cosine' or min_jerk",
    )
    parser.add_argument(
        "--input", type=str, help="data to streaming results", default="retarget_configs/data/1_smpl.pkl"
    )
    parser.add_argument(
        "--config", type=str, help="data to streaming results", default="retarget_configs/configs/smpl_gr1.yaml"
    )
    parser.add_argument("--save-video", action="store_true", default=False)
    args = parser.parse_args()
    return args


class Traj_Generator:
    def __init__(
        self,
        retargeter,
        upperbody_controller,
        interpolator="linear",
        interpolate_len=3,
        adjust_len=3,
        init_qpos=None,
        init_intepolate_steps=30,
    ):

        self.last_target = None
        self.next_target = None

        self.cur_step = 0
        self.cur_idx = 0
        self.interpolate_len = interpolate_len
        self.init_interpolate_steps = init_intepolate_steps
        self.adjust_len = adjust_len
        self.first_try = True

        if interpolator == "linear":
            self.interpolator = self.linear_interpolation
        elif interpolator == "cosine":
            self.interpolator = self.cosine_interpolation
        elif interpolator == "min_jerk":
            self.interpolator = self.min_jerk_interpolation

        self.retargeter = retargeter
        self.upperbody_controller = upperbody_controller

    def linear_interpolation(self):
        return self.last_target + (self.next_target - self.last_target) * (self.cur_step / self.interpolate_len)

    def cosine_interpolation(self):
        return (
            self.last_target
            + (self.next_target - self.last_target) * (1 - np.cos(np.pi * self.cur_step / self.interpolate_len)) / 2
        )

    def min_jerk_interpolation(self):
        return self.last_target + (self.next_target - self.last_target) * (
            10 * (self.cur_step / self.interpolate_len) ** 3
            - 15 * (self.cur_step / self.interpolate_len) ** 4
            + 6 * (self.cur_step / self.interpolate_len) ** 5
        )

    def ik_res_to_target_joint_pos(self, ik_joint_qpos):
        """
        Calculate the target joint positions from the results of inverse kinematics.

        Args:
            ik_joint_results: np array of shape (56, ), the results of inverse kinematics for all body joints
                            order: 3 waist + 3 head + 7 left arm + 12 left hand + 7 right arm + 12 right hand + 6 left leg + 6 right leg

        Returns:
            target_qpos: np array of shape (32, ), the target joint qpos for the robot
                        order: 3 waist + 3 head + 7 right arm + 7 left arm + 6 right hand + 6 left hand

        """
        target_qpos = np.zeros(32)  # 3 waist + 3 head + 7 right arm + 7 left arm + 6 right hand + 6 left hand
        target_qpos[0:6] = ik_joint_qpos[0:6]  # waist + head
        target_qpos[6:13] = ik_joint_qpos[25:32]  # right arm
        target_qpos[13:20] = ik_joint_qpos[6:13]  # left arm

        actuator_idxs = np.array([0, 1, 8, 10, 4, 6])
        target_qpos[20:26] = ik_joint_qpos[32 + actuator_idxs]  # right hand
        target_qpos[26:32] = ik_joint_qpos[13 + actuator_idxs]  # left hand
        return target_qpos

    def step(self):
        if self.cur_idx % self.cycle_len == 0:
            data_t = s[int(self.cur_idx / self.cycle_len)]
            ik_res, _, target_pos = self.retargeter(data_t, self.offset)
            if self.first_try:
                self.last_target = self.init_qpos
                self.first_try = False
            else:
                if self.interpolate_len == self.init_interpolate_steps:
                    self.cur_idx -= self.init_interpolate_steps - 3
                    self.interpolate_len = 3
                self.last_target = self.next_target.copy()
            self.next_target = ik_res
            self.next_target = self.ik_res_to_target_joint_pos(self.next_target)


class Upperbody_Controller:
    def __init__(self, config):
        pass

    def gripper_joint_pos_controller(self, obs, desired_qpos, kp=100, damping_ratio=1):
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

    def calculate_action_from_target_joint_pos(self, obs, target_joints):
        """
        Given target joint pos, calculate action for grippers and body joints.

        Args:
            obs: dict, the observation from the environment
            target_joints: np.array of shape (32, ), the target joint qpos for the robot.

        Returns:
            action: np.array of shape (32, ), the action for the robot
        """
        # order of actions: 3 waist + 3 head + 4 right arm + 6 right hand + 3 rigth arm + 7 left arm + 6 left hand
        action = np.zeros(32)

        # order of target joints: 3 waist + 3 head + 7 right arm + 7 left arm + 6 right hand + 6 left hand
        gripper_action = gripper_joint_pos_controller(obs, target_joints[-12:], kp=100)

        # right hand (action[10:16])
        action[10:16] = gripper_action[:6]
        # left hand (action[26:32])
        action[26:32] = gripper_action[6:]
        # waist + head + arms
        action[0:10] = np.clip(5 * (target_joints[0:10] - obs["robot0_joint_pos"][0:10]), -3, 3)
        action[16:26] = np.clip(5 * (target_joints[10:20] - obs["robot0_joint_pos"][10:20]), -3, 3)
        # special care for the head
        action[3:6] = np.clip(0.1 * (target_joints[3:6] - obs["robot0_joint_pos"][3:6]), -0.0, 0.0)

        return action

    def step(self):
        pass


class Retargeter:
    def __init__(self, config, data_path, offset={"link_RArm7": [0, 0, 0]}, vis=False):
        config = load_config(config)
        self.retargeter = SMPLGR1Retargeter(config, vis=vis)

        with open(data_path, "rb") as f:
            s = pickle.load(f)

        print("data len=", len(s))
        s = s[88 : 88 + 20]
        print("new data len=", len(s))

        data0 = s[0]
        self.retargeter.calibrate(data0)
        self.data = s

        with open("scripts/ik_weight_template.json", "r") as f:
            self.ik_weights = json.load(f)

        self.offset = offset

    def update_weights(self, new_weights):
        """
        Update IK weights for the retargeter to @new_weights.

        Args:
            new_weights: np.array of shape (6, ), the new weights for the IK solver.
                (order: L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist)
        """

        for i, link in enumerate(self.ik_weights["GR1_body"]):
            self.ik_weights["GR1_body"][link]["position_cost"] = new_weights[i]
        self.retargeter.update_weights(self.ik_weights)

    def ik(self, data_idx):
        return self.retargeter(self.data[data_idx], self.offset)

    def adjust(self, ik_weights, relative_trans):
        return self.retargeter.control(ik_weights, relative_trans)


def main():
    args = get_args()

    retargeter = Retargeter(args.config, args.input, offset={"link_RArm7": [0, 0, 0]})

    exp_weights = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    data = []
    headers = ["shoulder weight", "shoulder err", "elbow weight", "elbow err", "wrist weight", "wrist err"]

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
        camera_names=["frontview", "agentview"],
    )
    obs = env.reset()
    robot_offset = env.robots[0].robot_model.base_xpos_offset["table"](0.7)
    robot_offset += env.robots[0].robot_model.top_offset
    print("robot offset=", robot_offset)

    if args.save_video:
        # 1080p
        os.makedirs("tmp_video", exist_ok=True)
        writer = cv2.VideoWriter(f"tmp_video/adjust_{exp_idx}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (1080, 1080))

    mjmodel = env.sim.model._model
    mjdata = env.sim.data._data

    fps = 20
    with mujoco.viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:

        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
            viewer.cam.azimuth = 180
            viewer.cam.lookat = np.array([0.0, 0.0, 1.5])
            viewer.cam.distance = 1.0
            viewer.cam.elevation = -35

        idx = 0.0
        interpolate_len = 3
        adjust_len = 3

        next_ik_target = np.zeros(56)
        last_ik_target = np.zeros(56)
        first_try = True
        init_qpos = np.array(
            [
                -0.0035970834452681436,
                0.011031227286351492,
                -0.01311470003464996,
                0.0,
                0.0,
                0.0,
                0.8511509067155127,
                1.310805039853726,
                -0.7118440391862395,
                -0.536551596928798,
                0,
                0,
                0,
                0,
                0,
                0,
                0.02341464067352966,
                -0.23317144423063796,
                -0.0803808564555934,
                0.18086797377837605,
                -1.5034221574091646,
                -0.15101789788918812,
                0.00014316406250000944,
                -0.07930486850248092,
                -0.1222325540688668,
                -0.2801763429367678,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        target_pos = {}
        while viewer.is_running():

            if int(idx) % (interpolate_len + adjust_len) == 0:
                data_t = s[int(idx / (interpolate_len + adjust_len))]
                ik_res, _, target_pos = retargeter(data_t, offset)
                # print("target_pos=", target_pos)
                if first_try:
                    last_target = init_qpos
                    first_try = False
                else:
                    if interpolate_len == 30:
                        idx -= 27
                        interpolate_len = 3
                    last_ik_target = next_ik_target.copy()
                    last_target = calculate_target_qpos(last_ik_target)
                next_ik_target = ik_res
                next_target = calculate_target_qpos(next_ik_target)

            cur_step = int(idx) % (interpolate_len + adjust_len)
            idx += 1

            if cur_step < interpolate_len:
                if args.interpolation == "linear":
                    target_joints = linear_interpolation(last_target, next_target, cur_step, interpolate_len)
                elif args.interpolation == "cosine":
                    target_joints = cosine_interpolation(last_target, next_target, cur_step, interpolate_len)
                elif args.interpolation == "min_jerk":
                    target_joints = min_jerk_interpolation(last_target, next_target, cur_step, interpolate_len)
            else:

                trans = np.eye(4)
                left_eef_pos = obs["robot0_left_eef_pos"] - robot_offset
                target_left_eef_pos = target_pos["link_LArm7"][:3, 3]
                trans[:3, 3] = (target_left_eef_pos - left_eef_pos) * 0.1

                target_joints = calculate_target_qpos(
                    retargeter.control({"link_LArm7": ik_weights["GR1_body"]["link_LArm7"]}, trans)
                )

                trans = np.eye(4)
                right_eef_pos = obs["robot0_right_eef_pos"] - robot_offset
                target_right_eef_pos = target_pos["link_RArm7"][:3, 3]
                trans[:3, 3] = (target_right_eef_pos - right_eef_pos) * 0.1

                target_joints = calculate_target_qpos(
                    retargeter.control({"link_RArm7": ik_weights["GR1_body"]["link_RArm7"]}, trans)
                )

            action = calculate_action_from_target_joint_pos(obs, target_joints)
            obs, reward, done, _ = env.step(action)

            # save image
            img = obs["frontview_image"]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img[::-1, :, :]
            viewer.sync()
            cv2.imshow("image", img)
            cv2.waitKey(10)
            if args.save_video:
                # make the size 1080p, also drop the a channel
                img = cv2.resize(img, (1080, 1080))
                writer.write(img)

            # calculate error in tracking
            if idx % (interpolate_len + adjust_len) == 0:
                left_eef_pos = obs["robot0_left_eef_pos"] - robot_offset
                right_eef_pos = obs["robot0_right_eef_pos"] - robot_offset
                target_left_eef_pos = target_pos["link_LArm7"][:3, 3]
                target_right_eef_pos = target_pos["link_RArm7"][:3, 3]
                print("right eef pos=", right_eef_pos, "target right eef pos=", target_right_eef_pos)
                print("finish index", idx / (interpolate_len + adjust_len))
                error = {
                    "link_LArm7": np.linalg.norm(left_eef_pos - target_left_eef_pos),
                    "link_RArm7": np.linalg.norm(right_eef_pos - target_right_eef_pos),
                    "link_LArm2": 0,
                    "link_RArm2": 0,
                    "link_LArm4": 0,
                    "link_RArm4": 0,
                }
                for k in error.keys():
                    if k not in total_error:
                        total_error[k] = 0
                    total_error[k] += error[k]

            # check if an experiment of ik weights ends
            if idx >= len(s) * (interpolate_len + adjust_len):
                if args.save_video:
                    writer.release()
                    print(f"Saved video to tmp_video/{exp_idx}.mp4")

                # calculate error & save data
                for k in total_error.keys():
                    total_error[k] /= len(s)
                data.append(
                    [
                        ik_weights["GR1_body"]["link_RArm2"]["position_cost"],
                        total_error["link_RArm2"],
                        ik_weights["GR1_body"]["link_RArm4"]["position_cost"],
                        total_error["link_RArm4"],
                        ik_weights["GR1_body"]["link_RArm7"]["position_cost"],
                        total_error["link_RArm7"],
                    ]
                )
                print("exp_idx=", exp_idx, "finished")

                exp_idx += 1
                idx = 0.0

                if exp_idx >= len(exp_weights):
                    break

            time.sleep(1 / fps)

        print(tabulate.tabulate(data, headers=headers))


if __name__ == "__main__":
    main()
