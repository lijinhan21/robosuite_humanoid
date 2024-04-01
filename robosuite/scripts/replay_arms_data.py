import argparse
import json
import os
import time

import mujoco
import numpy as np
from mujoco import viewer

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="HumanoidTransport")
    parser.add_argument("--motion_id", type=str, default="000228")
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        help="Choice of interpolate algorithm. Can be 'linear' or 'cosine' or min_jerk",
    )
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

    data = np.load(f"data/motion/{args.motion_id}/{args.motion_id}_joints_cleaned.npy")
    # data = np.concatenate((np.zeros((1, 39)), data[0:]), axis=0)
    print(data.shape, len(data), np.max(data), np.min(data))

    mjmodel = env.sim.model._model
    mjdata = env.sim.data._data

    saved_data = []

    fps = 30
    cycle_len = 8  # 30
    repeat_len = 1  # 100
    with mujoco.viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        idx = 0.0
        repeat_idx = 0
        while viewer.is_running():
            # action for joints position controller: 7 right arm + 6 right hand + 7 left arm + 6 left hand
            action = np.zeros(26)

            next_idx = (np.floor(idx / cycle_len + 1).astype(int)) % len(data)
            last_idx = (np.floor(idx / cycle_len).astype(int)) % len(data)
            next_target = data[next_idx][13:27]  # 7 left arm + 7 right hand
            last_target = data[last_idx][13:27]  # 7 left arm + 7 right hand
            cur_step = int(idx) % cycle_len

            repeat_idx += 1
            if repeat_idx % repeat_len == 0:
                repeat_idx = 0
                idx += 1

            if args.interpolation == "linear":
                target_joints = linear_interpolation(last_target, next_target, cur_step, cycle_len)
            elif args.interpolation == "cosine":
                target_joints = cosine_interpolation(last_target, next_target, cur_step, cycle_len)
            elif args.interpolation == "min_jerk":
                target_joints = min_jerk_interpolation(last_target, next_target, cur_step, cycle_len)

            # # for debugging
            # target_joints *= 0
            # target_joints[0] = -1
            # target_joints[7] = 1
            # target_joints = data[0][13:27]

            # action: 7 right arm + 6 right hand + 7 left arm + 6 left hand
            # target_joints: 7 left arm + 7 right hand
            # obs["robot0_joint_pos"]: 7 right arm + 7 left arm
            action[13:20] = np.clip(5 * (target_joints[0:7] - obs["robot0_joint_pos"][7:14]), -3, 3)
            action[0:7] = np.clip(5 * (target_joints[7:14] - obs["robot0_joint_pos"][0:7]), -3, 3)

            # print("delta=", target_joints[0:7] - obs["robot0_joint_pos"][7:14], target_joints[7:14] - obs["robot0_joint_pos"][0:7])
            # print("target=", target_joints[0:14])
            # print("current=", obs["robot0_joint_pos"][7:14], obs["robot0_joint_pos"][0:7])

            max_delta = max(
                np.max(np.abs(target_joints[0:5] - obs["robot0_joint_pos"][7:12])),
                np.max(np.abs(target_joints[7:12] - obs["robot0_joint_pos"][0:5])),
            )
            print("idx=", idx, next_idx, last_idx, cur_step, max_delta)
            print(target_joints[0:7], "|||||||||||", obs["robot0_joint_pos"][7:14])
            print(target_joints[7:14], "|||||||||||", obs["robot0_joint_pos"][0:7])

            obs, reward, done, _ = env.step(action)

            saved_data.append(
                {
                    "target_joints": np.concatenate((target_joints[7:14], target_joints[0:7])).tolist(),
                    "actual_joints": obs["robot0_joint_pos"].tolist(),
                }
            )

            viewer.sync()
            time.sleep(1 / fps)

            if idx > len(data) * cycle_len:
                break

    with open(f"data/motion/{args.motion_id}/replay_data_{args.interpolation}.json", "w", encoding="utf-8") as file_obj:
        json.dump(saved_data, file_obj, ensure_ascii=False)
