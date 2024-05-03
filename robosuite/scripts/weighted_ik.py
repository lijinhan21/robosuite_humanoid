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
import torch
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
        "--input", type=str, help="data to streaming results", default="retarget_configs/data/7_smpl.pkl"
    )
    parser.add_argument(
        "--tap_res",
        type=str,
        help="data to streaming tap results",
        default="retarget_configs/data/7_tap_temporal_segments.pt",
    )
    parser.add_argument(
        "--config", type=str, help="data to streaming results", default="retarget_configs/configs/smpl_gr1.yaml"
    )
    parser.add_argument("--save-video", action="store_true", default=False)
    args = parser.parse_args()
    return args


def linear_interpolation(last_target, next_target, cur_step, cycle_len):
    return last_target + (next_target - last_target) * (cur_step / cycle_len)


def cosine_interpolation(last_target, next_target, cur_step, cycle_len):
    return last_target + (next_target - last_target) * (1 - np.cos(np.pi * cur_step / cycle_len)) / 2


def min_jerk_interpolation(last_target, next_target, cur_step, cycle_len):
    return last_target + (next_target - last_target) * (
        10 * (cur_step / cycle_len) ** 3 - 15 * (cur_step / cycle_len) ** 4 + 6 * (cur_step / cycle_len) ** 5
    )


def calculate_target_qpos(ik_joint_qpos):
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


def convert_from_target_qpos_to_ik_joint_qpos(target_qpos):
    ik_pos = np.zeros(56)
    ik_pos[0:6] = target_qpos[0:6]
    ik_pos[25:32] = target_qpos[6:13]
    ik_pos[6:13] = target_qpos[13:20]

    actuator_idxs = np.array([0, 1, 8, 10, 4, 6])
    ik_pos[32 + actuator_idxs] = target_qpos[20:26]
    ik_pos[13 + actuator_idxs] = target_qpos[26:32]
    return ik_pos


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


def calculate_action_from_target_joint_pos(obs, target_joints):
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


def main():
    args = get_args()
    config = load_config(args.config)
    # s = np.load(args.input).astype(np.float64)  # T 52 4 4

    retargeter = SMPLGR1Retargeter(config, vis=True)

    with open(args.input, "rb") as f:
        s = pickle.load(f)
    with open(args.tap_res, "rb") as f:
        tap_res = torch.load(f)

    print("data len=", len(s))
    print("len of tap_res=", len(tap_res), tap_res)
    # rescale tap_res to fit the length of s
    len_tap_res = tap_res[-1][1]
    p = len(s) / len_tap_res
    important_indices = [0]
    steps_btw_keyframes = [3, 20, 3]
    for i in range(len(tap_res)):
        last_idx = important_indices[-1]
        new_idx = math.floor((tap_res[i][1] + 1) * p - 1)
        for j in range(steps_btw_keyframes[i]):
            important_indices.append(math.floor(last_idx + (new_idx - last_idx) * (j + 1) / steps_btw_keyframes[i]))
        # important_indices.append(math.floor((tap_res[i][1] + 1) * p - 1))
    print("important_indices=", important_indices)
    s = [s[i] for i in important_indices]
    # s = s[88:88 + 20] # Note: for testing grasp on 1_smpl (play with boat)
    print("new data len=", len(s))

    # exit(0)

    data0 = s[0]

    retargeter.calibrate(data0)
    print("Calibrated, start streaming...")

    with open("scripts/ik_weight_template.json", "r") as f:
        ik_weights = json.load(f)

    exp_idx = 0
    exp_weights = [
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]
    data = []
    headers = ["shoulder weight", "shoulder err", "elbow weight", "elbow err", "wrist weight", "wrist err"]

    offsets = [
        {
            "link_RArm7": [0.0, 0.0, 0.0],
        },
        {
            "link_RArm7": [0.05, 0.1, 0.0],
        },
        {
            "link_RArm7": [0.0, 0.1, -0.05],
        },
        {
            "link_RArm7": [0.0, -0.05, 0.1],
        },
        {
            "link_RArm7": [0.05, -0.05, 0.05],
        },
        {
            "link_RArm7": [0.0, 0.2, -0.1],
        },
        {
            "link_RArm7": [-0.05, -0.0, 0.05],
        },
        {
            "link_RArm7": [-0.05, -0.0, 0.1],
        },
        {
            "link_RArm7": [0.05, 0.05, -0.05],
        },
    ]
    offset_idx = 0
    offset = offsets[offset_idx]

    config = {}
    config["robots"] = "GR1UpperBody"
    config["env_name"] = args.environment
    config["env_configuration"] = "bimanual"
    config["controller_configs"] = load_controller_config(default_controller="JOINT_POSITION")
    config["controller_configs"]["kp"] = 500
    config["retarget_offsets"] = offset["link_RArm7"]

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
    robot_offset = env.robots[0].robot_model.base_xpos_offset["table"](0.8)
    robot_offset += env.robots[0].robot_model.top_offset
    print("robot offset=", robot_offset)

    mjmodel = env.sim.model._model
    mjdata = env.sim.data._data

    saved_joint_pos = []

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
        init_interpolate_steps = 30
        normal_interpolate_steps = 10
        in_init_steps = True
        interpolate_len = 3
        adjust_len = 0

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

            # check if a new experiment of ik weights starts
            if idx < 1 and idx > -1:
                if args.save_video:
                    # 1080p
                    os.makedirs("tmp_video", exist_ok=True)
                    writer = cv2.VideoWriter(
                        f"tmp_video/{offset_idx}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (1080, 1080)
                    )
                    # time.sleep(1)

                for i, link in enumerate(ik_weights["GR1_body"]):
                    ik_weights["GR1_body"][link]["position_cost"] = exp_weights[exp_idx][i]
                retargeter.update_weights(ik_weights)
                total_error = {}

                # obs = env.reset()
                robot_offset = env.robots[0].robot_model.base_xpos_offset["table"](0.8)
                robot_offset += env.robots[0].robot_model.top_offset
                print("robot offset=", robot_offset)

                interpolate_len = 30
                first_try = True

            if int(idx) % (interpolate_len + adjust_len) == 0:
                data_t = s[int(idx / (interpolate_len + adjust_len))]
                ik_res, _, target_pos = retargeter(data_t, offset)
                # print("target_pos=", target_pos)
                if first_try:
                    last_target = init_qpos
                    first_try = False
                else:
                    if in_init_steps == True:
                        idx -= init_interpolate_steps - normal_interpolate_steps
                        interpolate_len = normal_interpolate_steps
                        in_init_steps = False
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

            saved_joint_pos.append(convert_from_target_qpos_to_ik_joint_qpos(target_joints))

            action = calculate_action_from_target_joint_pos(obs, target_joints)
            obs, reward, done, _ = env.step(action)

            # calculate pos error
            left_eef_pos = obs["robot0_left_eef_pos"] - robot_offset
            right_eef_pos = obs["robot0_right_eef_pos"] - robot_offset
            target_left_eef_pos = target_pos["link_LArm7"][:3, 3]
            target_right_eef_pos = target_pos["link_RArm7"][:3, 3]
            left_eef_error = target_left_eef_pos - left_eef_pos
            right_eef_error = target_right_eef_pos - right_eef_pos

            # save image
            img = obs["frontview_image"]
            img = cv2.flip(img, 0)
            cv2.putText(
                img,
                f"err=[{right_eef_error[0]:.2f}, {right_eef_error[1]:.2f}, {right_eef_error[2]:.2f}] norm={np.linalg.norm(right_eef_error):.2f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
            # cv2.putText(img, f'''offset=[{offset["link_RArm7"][0]:.2f}, {offset["link_RArm7"][1]:.2f}, {offset["link_RArm7"][2]:.2f}]''', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            viewer.sync()
            cv2.imshow("image", img)
            cv2.waitKey(10)
            if args.save_video:
                # make the size 1080p, also drop the a channel
                img = cv2.resize(img, (1080, 1080))
                writer.write(img)

            # calculate error in tracking
            if idx % (interpolate_len + adjust_len) == 0:
                print("right eef pos=", right_eef_pos, "target right eef pos=", target_right_eef_pos)
                print("finish index", idx / (interpolate_len + adjust_len))
                error = {
                    "link_LArm7": np.linalg.norm(left_eef_error),
                    "link_RArm7": np.linalg.norm(right_eef_error),
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

    saved_joint_pos = np.array(saved_joint_pos)
    np.save("7_keyframe_interpolate_results.npy", saved_joint_pos)


if __name__ == "__main__":
    main()
