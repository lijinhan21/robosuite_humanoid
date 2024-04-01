import argparse

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_id", type=str, default="000228")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    data = np.load(f"{args.motion_id}/{args.motion_id}_joints_update.npy")

    res = []
    for i in range(len(data)):
        if np.max(np.abs(data[i])) >= 3:
            continue
        if i % 2 == 0:
            res.append(data[i].tolist())

    np.save(f"{args.motion_id}/{args.motion_id}_joints_cleaned.npy", np.array(res))
