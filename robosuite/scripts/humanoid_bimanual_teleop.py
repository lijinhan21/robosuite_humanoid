"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


def collect_bimanual_trajectory(env, device_l, device_r, arm, env_configuration):
    """
    Use the two devices (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device_l.start_control()
    device_r.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        action_l, grasp_l = input2action(
            device=device_l, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )
        action_r, grasp_r = input2action(
            device=device_r, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )

        # If action is none, then this a reset so we should break
        if (action_l is None) or (action_r is None):
            break

        if env_configuration == "bimanual":
            # action = np.concatenate([np.zeros_like(action), action])
            # action_r = action
            # action_l = action.copy()
            # for idx in [1, 3, 5]:
            #     action_l[idx] *= -1

            # action_l[3: 6] = 0
            # action_r[3: 6] = 0
            action = np.concatenate([action_r, action_l])

        # Run environment step
        env.step(action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="HumanoidLift")
    parser.add_argument(
        "--robots", nargs="+", type=str, default="GR1UpperBody", help="Which robot(s) to use in the env"
    )
    parser.add_argument(
        "--config", type=str, default="bimanual", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument(
        "--device", type=str, default="spacemouse", help="Choice of device. Can be 'keyboard' or 'spacemouse'"
    )
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    controller_config["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
    controller_config["controller_configs"]["input_max"] = [50] * 6
    controller_config["controller_configs"]["input_min"] = [-50] * 6
    controller_config["controller_configs"]["output_max"] = [50] * 6
    controller_config["controller_configs"]["output_min"] = [-50] * 6
    controller_config["controller_configs"]["kp"] = 5000
    controller_config["controller_configs"]["control_delta"] = False  # True

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard" or args.device == "spacemouse":
        from robosuite.devices import T265, Keyboard, SpaceMouse

        device_l = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        # device_l = T265(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        device_r = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_bimanual_trajectory(env, device_l, device_r, args.arm, args.config)
