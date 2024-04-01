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
    args = parser.parse_args()
    return args


saved_data = []


def joint_pos_controller(obs, desired_qpos, kp=100, damping_ratio=1):
    """
    Calculate the torques for the joints position controller.

    Args:
        obs: dict, the observation from the environment
        desired_qpos: np.array of shape (12, ), right hand first, then left hand
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

    saved_data.append(
        {
            "torques": desired_torque.tolist(),
            "position_error": position_error.tolist(),
            "target_position": desired_qpos.tolist(),
            "actual_position": joint_qpos.tolist(),
            "vel_error": vel_pos_error.tolist(),
        }
    )
    return desired_torque


def angle_between(p1, p2, p3):
    """
    Calculate the angle between three points.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p1 - p2
    v2 = p3 - p2
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def project_to_plane(p, n, p0):
    """
    Project point p to the plane defined by normal vector n (note: n may not be of norm 1) and point p0.
    """
    p = np.array(p)
    n = np.array(n)
    n = n / np.linalg.norm(n)
    p0 = np.array(p0)
    return p - (np.dot(p - p0, n) / np.dot(n, n)) * n


def calculate_target_pos(joint_locations):
    """
    Given the 15 locations of keypoints on one hand, calculate the target joint position for the grippers.
    """
    target_joint_pos = np.zeros(6)

    # thumb_proximal_1
    proj_thumb = project_to_plane(
        joint_locations[14], np.array(joint_locations[0]) - np.array(joint_locations[12]), joint_locations[12]
    )
    proj_palm = np.mean(
        np.array(
            [
                project_to_plane(
                    joint_locations[3],
                    np.array(joint_locations[0]) - np.array(joint_locations[12]),
                    joint_locations[12],
                ),
                project_to_plane(
                    joint_locations[9],
                    np.array(joint_locations[0]) - np.array(joint_locations[12]),
                    joint_locations[12],
                ),
                project_to_plane(
                    joint_locations[6],
                    np.array(joint_locations[0]) - np.array(joint_locations[12]),
                    joint_locations[12],
                ),
            ]
        ),
        axis=0,
    )
    thumb_proximal_1 = np.pi - angle_between(proj_thumb, joint_locations[12], proj_palm)
    target_joint_pos[0] = thumb_proximal_1

    # thumb_proximal_2
    thumb_distal = np.pi - angle_between(joint_locations[12], joint_locations[13], joint_locations[14])
    thumb_proximal_2 = thumb_distal / 1.13
    target_joint_pos[1] = thumb_proximal_2

    # index_proximal
    index_distal = np.pi - angle_between(joint_locations[0], joint_locations[1], joint_locations[2])
    index_proximal = index_distal / 1.13
    target_joint_pos[2] = index_proximal

    # middle_proximal
    middle_distal = np.pi - angle_between(joint_locations[3], joint_locations[4], joint_locations[5])
    middle_proximal = middle_distal / 1.13
    target_joint_pos[3] = middle_proximal

    # ring_proximal
    ring_distal = np.pi - angle_between(joint_locations[9], joint_locations[10], joint_locations[11])
    ring_proximal = ring_distal / 1.08
    target_joint_pos[4] = ring_proximal

    # pinck_proximal
    pinky_distal = np.pi - angle_between(joint_locations[6], joint_locations[7], joint_locations[8])
    ring_proximal = pinky_distal / 1.15
    target_joint_pos[5] = ring_proximal

    return target_joint_pos


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
    config["controller_configs"]["kp"] = 300

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

    # read reference hand joints locations and calculate the target joint positions
    with open("hand_joints.json", "r") as f:
        joint_locations = json.load(f)
    replay_data = []
    for i in range(len(joint_locations)):
        replay_data.append(
            np.concatenate(
                (
                    calculate_target_pos(joint_locations[i][15:]),  # right hand
                    calculate_target_pos(joint_locations[i][:15]),
                )
            )  # left hand
        )

    # mujoco.viewer.launch(mjmodel, mjdata)
    # exit()
    fps = 20
    renderer = mujoco.Renderer(model=mjmodel, height=480, width=640)

    with mujoco.viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=True,
        show_right_ui=False,  # True,
    ) as viewer:

        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
            viewer.cam.azimuth = 180
            viewer.cam.lookat = np.array([0.0, 0.0, 1.5])
            viewer.cam.distance = 0.05
            viewer.cam.elevation = -45

        idx = 0.0
        cycle_len = 1
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
            # gripper_action = joint_pos_controller(obs, np.array([0.5, 0.2, 0.3, 0.6, 1.2, 0.8, 0, 0.5, 0, 0.1, 1.6, 0.7]), kp=100)
            gripper_action = joint_pos_controller(obs, target_joints, kp=100)

            # right hand (action[7:13])
            action[7:13] = gripper_action[:6]

            # left hand (action[20:26])
            action[20:26] = gripper_action[6:]

            obs, reward, done, _ = env.step(action)

            viewer.sync()
            time.sleep(1 / fps)

            # input()

            if idx % (len(replay_data) * cycle_len) == 0:
                with open("data_track_5cos.json", "w") as f:
                    json.dump(saved_data, f)
                break
