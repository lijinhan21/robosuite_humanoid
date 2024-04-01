import argparse
import time

import mujoco
from mujoco import viewer

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}
    options["env_name"] = "TwoArmLift"
    options["env_configuration"] = "bimanual"
    options["robots"] = "Baxter"

    # options["env_name"] = "HumanoidTransport"
    # options["env_configuration"] = 'bimanual'
    # options["robots"] = "GR1"

    options["env_name"] = "HumanoidReturnBook"  # "HumanoidHCI" #"HumanoidReturnBook" # #"HumanoidToyAssembly"
    options["env_configuration"] = "bimanual"
    options["robots"] = "GR1UpperBody"

    # OSC_POSE, JOINT_POSITION, JOINT_TORQUE
    # options["controller_configs"] = load_controller_config(default_controller="JOINT_TORQUE")
    # options["controller_configs"]["input_max"] = 20000
    # options["controller_configs"]["input_min"] = -20000
    # options["controller_configs"]["output_max"] = 20000
    # options["controller_configs"]["output_min"] = -20000
    # options["controller_configs"]["damping_ratio"] = 2

    # options["controller_configs"] = load_controller_config(default_controller="JOINT_POSITION")
    # options["controller_configs"]["input_max"] = 2
    # options["controller_configs"]["input_min"] = -2
    # options["controller_configs"]["output_max"] = 2
    # options["controller_configs"]["output_min"] = -2
    # options["controller_configs"]["damping_ratio"] = 2
    # options["controller_configs"]["kp"] = 1

    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
    options["controller_configs"]["input_max"] = [100] * 6
    options["controller_configs"]["input_min"] = [-100] * 6
    options["controller_configs"]["output_max"] = [100] * 6
    options["controller_configs"]["output_min"] = [-100] * 6
    options["controller_configs"]["kp"] = 500
    options["controller_configs"]["control_delta"] = False

    # options["controller_configs"] = load_controller_config(default_controller="OSC_POSITION")
    # options["controller_configs"]["input_max"] = [100] * 3
    # options["controller_configs"]["input_min"] = [-100] * 3
    # options["controller_configs"]["output_max"] = [100] * 3
    # options["controller_configs"]["output_min"] = [-100] * 3
    # options["controller_configs"]["kp"] = 500
    # options["controller_configs"]["control_delta"] = False

    print(options["controller_configs"])
    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    obs = env.reset()

    mjmodel = env.sim.model._model
    mjdata = env.sim.data._data

    # Get action limits
    low, high = env.action_spec
    print("shape", low.shape)

    fps = 25
    flag = 0

    mujoco.viewer.launch(model=mjmodel, data=mjdata)
    exit()

    # for joint torque
    # action_seq = np.zeros((100, 26))
    # action_seq[1][1] = 80
    # action_seq[1][3] = 30

    # action_seq[2][0] = -100
    # action_seq[2][1] = 40

    # action_seq[3][0] = -100
    # action_seq[3][1] = 30
    # action_seq[3][3] = 60

    # action_seq[4][0] = -100
    # action_seq[4][1] = 30
    # action_seq[4][3] = 60
    # action_seq[4][21:26] = -0.05

    # action_seq[5][0] = -70
    # action_seq[5][1] = 30
    # action_seq[5][3] = 70
    # action_seq[5][20] = 0.8
    # action_seq[5][21:26] = 0.05

    # action_seq[6][0] = -70
    # action_seq[6][1] = 20
    # action_seq[6][3] = 70
    # action_seq[6][20] = 0.8
    # action_seq[6][21:26] = 0.05

    # action_seq[7][0] = -20
    # action_seq[7][1] = -40
    # action_seq[7][3] = 50

    # action_seq[8][0] = -20
    # action_seq[8][1] = -40
    # action_seq[8][3] = 50

    # for osc pose
    action_seq = np.zeros((100, 24))
    action_seq[0][12:15] = np.array([-0.5, 0.8, 1.2])
    action_seq[0][18:24] = -1

    action_seq[1][12:15] = np.array([-0.5, 0.5, 1.5])
    action_seq[1][18:24] = -1

    action_seq[2][12:15] = np.array([-0.2, 0.4, 1.3])
    action_seq[2][15:18] = np.array([1.57, 0, 0])
    action_seq[2][18:24] = -1

    action_seq[3][12:15] = np.array([-0.35, 0.20, 1.35])
    action_seq[3][15:18] = np.array([1.57, 0, 0])
    action_seq[3][18:24] = -1

    action_seq[4][12:15] = np.array([-0.35, 0.20, 1.35])
    action_seq[4][15:18] = np.array([1.57, 0, 0])
    action_seq[4][18:24] = -1

    action_seq[5][12:15] = np.array([-0.35, 0.20, 1.2])
    action_seq[5][15:18] = np.array([3.14, 0, 0])
    action_seq[5][18:24] = -1

    action_seq[6][12:15] = np.array([-0.35, 0.20, 0.9])
    action_seq[6][15:18] = np.array([3.14, 0, 0])
    action_seq[6][18:24] = -1

    action_seq[7][12:15] = np.array([-0.35, 0.25, 0.83])
    action_seq[7][15:18] = np.array([3.14, 0, 0])
    action_seq[7][18:24] = -1

    action_seq[8][12:15] = np.array([-0.35, 0.35, 0.83])
    action_seq[8][15:18] = np.array([3.14, 0, 0])
    action_seq[8][18:24] = 0

    action_seq[9][12:15] = np.array([-0.35, 0.35, 0.83])
    action_seq[9][15:18] = np.array([3.14, 0, 0])
    action_seq[9][18:24] = 0.3

    action_seq[10][12:15] = np.array([-0.35, 0.35, 0.83])
    action_seq[10][15:18] = np.array([3.14, 0, 0])
    action_seq[10][18:24] = 0.8

    action_seq[11][12:15] = np.array([-0.35, 0.35, 0.93])
    action_seq[11][15:18] = np.array([3.14, 0, 0])
    action_seq[11][18:24] = 0.8

    action_seq[12][12:15] = np.array([-0.35, 0.35, 0.95])
    action_seq[12][15:18] = np.array([3.14, 1, 1])
    action_seq[12][18:24] = 1.0

    action_seq[13][12:15] = np.array([-0.35, 0.30, 1.0])
    action_seq[13][15:18] = np.array([3.14, 0, 0])
    action_seq[13][18:24] = 1.0

    action_seq[14][12:15] = np.array([-0.35, 0.27, 1.0])
    action_seq[14][15:18] = np.array([3.14, 1, 1])
    action_seq[14][18:24] = 1.0

    cycle_len = 25
    with mujoco.viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        idx = 0.0
        while viewer.is_running():
            # action = np.random.uniform(low, high)
            action = np.zeros_like(low)

            # joints torque: 7 left arm + 6 right hand + 7 right arm + 6 left hand
            # osc pose: 6 left eef (pos + ori) + 6 right hand + 6 right eef (pos + ori) + 6 left hand

            idx += 1
            num = (np.floor(idx / cycle_len) + 1).astype(int)
            print("num =", num)
            for i in range(len(low)):
                action[i] = action_seq[num - 1][i] * (1 - idx % int(cycle_len) / cycle_len) + action_seq[num][i] * (
                    idx % int(cycle_len) / cycle_len
                )

            obs, reward, done, _ = env.step(action)

            viewer.sync()
            time.sleep(1 / fps)
