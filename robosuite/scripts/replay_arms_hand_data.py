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
    actuator_idxs = [0, 1, 5, 7, 9, 11]
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
                          order: 6 left leg + 6 right leg + 3 waist + 3 head + 7 left arm + 12 left hand + 7 right arm + 12 right hand

    Returns:
        target_qpos: np array of shape (26, ), the target joint qpos for the robot
                     order: 7 right arm + 7 left arm + 6 right hand + 6 left hand

    """
    target_qpos = np.zeros(26)  # 7 right arm + 7 left arm + 6 right hand + 6 left hand
    target_qpos[0:7] = ik_joint_qpos[37:44]  # right arm
    target_qpos[7:14] = ik_joint_qpos[18:25]  # left arm

    actuator_idxs = np.array([0, 1, 9, 11, 5, 7])
    target_qpos[14:20] = ik_joint_qpos[44 + actuator_idxs]  # right hand
    target_qpos[20:26] = ik_joint_qpos[25 + actuator_idxs]  # left hand
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
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
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

    fps = 20
    with mujoco.viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=True,
        show_right_ui=False,  # True,
    ) as viewer:

        idx = 0.0
        cycle_len = 5
        while viewer.is_running():
            # action for joints position controller: 7 right arm + 6 right hand + 7 left arm + 6 left hand
            action = np.zeros(26)

            next_idx = (np.floor(idx / cycle_len + 1).astype(int)) % len(replay_data)
            last_idx = (np.floor(idx / cycle_len).astype(int)) % len(replay_data)
            next_target = replay_data[next_idx]  # 6 right hand + 6 left hand
            last_target = replay_data[last_idx]  # 6 right hand + 6 left hand
            cur_step = int(idx) % cycle_len
            idx += 1

            if args.interpolation == "linear":
                target_joints = linear_interpolation(last_target, next_target, cur_step, cycle_len)
            elif args.interpolation == "cosine":
                target_joints = cosine_interpolation(last_target, next_target, cur_step, cycle_len)
            elif args.interpolation == "min_jerk":
                target_joints = min_jerk_interpolation(last_target, next_target, cur_step, cycle_len)

            # joint limits [0, 1.3], [0, 0.68], [0, 1.62], [0, 1.62], [0, 1.62], [0, 1.62]
            gripper_action = gripper_joint_pos_controller(obs, target_joints[-12:], kp=100)

            # right hand (action[7:13])
            action[7:13] = gripper_action[:6]
            # left hand (action[20:26])
            action[20:26] = gripper_action[6:]
            # right arm (action[0:7])
            action[0:7] = np.clip(5 * (target_joints[0:7] - obs["robot0_joint_pos"][0:7]), -3, 3)
            # left arm (action[13:20])
            action[13:20] = np.clip(5 * (target_joints[7:14] - obs["robot0_joint_pos"][7:14]), -3, 3)

            obs, reward, done, _ = env.step(action)

            actuator_idxs = [0, 1, 5, 7, 9, 11]
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
                with open("data_arm+hand_track.json", "w") as f:
                    json.dump(saved_data, f)
                break
