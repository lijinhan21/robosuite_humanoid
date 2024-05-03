import argparse
import json
import os
import time

import cv2
import mujoco
import numpy as np
from mujoco import viewer

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="HumanoidTransport")
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        help="Choice of interpolate algorithm. Can be 'linear' or 'cosine' or min_jerk",
    )
    parser.add_argument("--motion_id", type=str, default="000003")
    args = parser.parse_args()
    return args


saved_data = []


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


def linear_interpolation(last_target, next_target, cur_step, cycle_len):
    return last_target + (next_target - last_target) * (cur_step / cycle_len)


def cosine_interpolation(last_target, next_target, cur_step, cycle_len):
    return last_target + (next_target - last_target) * (1 - np.cos(np.pi * cur_step / cycle_len)) / 2


def min_jerk_interpolation(last_target, next_target, cur_step, cycle_len):
    return last_target + (next_target - last_target) * (
        10 * (cur_step / cycle_len) ** 3 - 15 * (cur_step / cycle_len) ** 4 + 6 * (cur_step / cycle_len) ** 5
    )


if __name__ == "__main__":
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
        camera_names=["frontview", "agentview"],
    )
    obs = env.reset()

    mjmodel = env.sim.model._model
    mjdata = env.sim.data._data

    # read ik joints qpos and calculate the target joint qpos
    raw_data = np.load(f"data/motion/{args.motion_id}/{args.motion_id}_ik.npy")
    replay_data = []
    for i in range(len(raw_data)):
        replay_data.append(calculate_target_qpos(raw_data[i]))
    print(len(replay_data), len(raw_data), raw_data.shape)

    writer = cv2.VideoWriter(f"tmp_video/adjust_.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (1080, 1080))

    fps = 20
    with mujoco.viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=False,  # True,
        show_right_ui=False,  # True,
    ) as viewer:

        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
            viewer.cam.azimuth = 180
            viewer.cam.lookat = np.array([0.0, 0.0, 1.5])
            viewer.cam.distance = 1.0
            viewer.cam.elevation = -35

        idx = 0.0
        cycle_len = 3
        while viewer.is_running():
            # joints order:
            #'robot0_waist_yaw', 'robot0_waist_pitch', 'robot0_waist_roll', 'robot0_head_yaw', 'robot0_head_roll', 'robot0_head_pitch', 'robot0_r_shoulder_pitch', 'robot0_r_shoulder_roll', 'robot0_r_shoulder_yaw', 'robot0_r_elbow_pitch', 'robot0_r_wrist_yaw', 'robot0_r_wrist_roll', 'robot0_r_wrist_pitch', 'robot0_l_shoulder_pitch', 'robot0_l_shoulder_roll', 'robot0_l_shoulder_yaw', 'robot0_l_elbow_pitch', 'robot0_l_wrist_yaw', 'robot0_l_wrist_roll', 'robot0_l_wrist_pitch'

            # action for joints position controller for upper body:
            # 3 waist + 3 head + 4 right arm + 6 right hand + 3 rigth arm + 7 left arm + 6 left hand
            action = np.zeros(32)

            next_idx = (np.floor(idx / cycle_len + 1).astype(int)) % len(replay_data)
            last_idx = (np.floor(idx / cycle_len).astype(int)) % len(replay_data)
            next_target = replay_data[next_idx]
            last_target = replay_data[last_idx]
            cur_step = int(idx) % cycle_len
            idx += 1

            # geom_id = env.sim.model.geom_name2id("robot0_left_wrist_target")
            # print("geom_id", geom_id, env.sim.data.geom_xpos[geom_id])
            # env.sim.data.geom_xpos[geom_id][1] = idx * 0.05
            # # env.sim.step()

            if args.interpolation == "linear":
                target_joints = linear_interpolation(last_target, next_target, cur_step, cycle_len)
            elif args.interpolation == "cosine":
                target_joints = cosine_interpolation(last_target, next_target, cur_step, cycle_len)
            elif args.interpolation == "min_jerk":
                target_joints = min_jerk_interpolation(last_target, next_target, cur_step, cycle_len)

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

            obs, reward, done, _ = env.step(action)
            # print("obs keys", obs.keys())
            # print(obs["frontview_image"].shape)

            # save image
            img = obs["frontview_image"]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", img)
            cv2.waitKey(1)

            actuator_idxs = [0, 1, 4, 6, 8, 10]
            saved_data.append(
                {
                    "target_position": target_joints.tolist(),
                    "actual_position": np.concatenate(
                        [
                            obs["robot0_joint_pos"],
                            obs["robot0_right_gripper_qpos"][actuator_idxs],
                            obs["robot0_left_gripper_qpos"][actuator_idxs],
                        ]
                    ).tolist(),
                }
            )

            viewer.sync()
            time.sleep(1 / fps)

            # input()

            if idx % (len(replay_data) * cycle_len) == 0:
                with open(f"data{args.motion_id}_arm+hand_track.json", "w") as f:
                    json.dump(saved_data, f)
                break
