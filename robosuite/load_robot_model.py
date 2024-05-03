import argparse
import time

import mujoco
import numpy as np
from mujoco import viewer
from retarget.retargeter import TeleopAVPGR1Retargeter
from retarget.streamer import AVPStreamer
from retarget.streamer.avp_record_streamer import AVPRecordStreamer
from retarget.utils.configs import load_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="HumanoidTransport")
    parser.add_argument("--ip", type=str, help="AVP IP address")
    parser.add_argument("--input", type=str, default="data/eg_clean_data.pkl", help="data to streaming results")
    args = parser.parse_args()
    return args


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


if __name__ == "__main__":
    args = get_args()

    config = load_config("configs/teleop_avp_gr1.yaml")
    retargeter = TeleopAVPGR1Retargeter(config, vis=False)

    s = AVPRecordStreamer(path=args.input)
    # s = AVPStreamer(args.ip)
    data0 = s.get()
    retargeter.calibrate(data0)
    print("Calibrated, start streaming...")

    mjmodel = mujoco.MjModel.from_xml_path("exported_robot_model.xml")

    start_time = time.time()
    mjdata = mujoco.MjData(mjmodel)
    print("Time to load model: ", time.time() - start_time)
    print("qpos", mjdata.qpos, len(mjdata.qpos))

    # viewer.launch(mjmodel, mjdata)

    # exit()

    with viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=False,  # True,
        show_right_ui=False,  # True,
    ) as viewer:
        while viewer.is_running():
            # update joint states of mjdata

            viewer.opt.geomgroup[0] = 0
            # mjdata.qpos[:] = np.random.rand(mjmodel.nq)

            try:
                data_t = s.get()
            except IndexError:
                s.reset()
                print("reset")
                data_t = s.get()
            # data_t = s.get()

            ik_res = retargeter(data_t)
            target_joints = calculate_target_qpos(ik_res)
            # order of target joints: 3 waist + 3 head + 7 right arm + 7 left arm + 6 right hand + 6 left hand
            # mjdata.qpos[:] = 0

            # mujoco.mj_step(mjmodel, mjdata, 1)

            # update the model
            viewer.sync()
            time.sleep(1 / 60)
