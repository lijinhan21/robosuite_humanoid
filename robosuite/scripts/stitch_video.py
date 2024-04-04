import argparse

import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_id", type=str, default="000003")
    args = parser.parse_args()
    return args


def stitch_videos(robot_video_path, human_video_path, output_path):
    cap = cv2.VideoCapture(robot_video_path)
    robot_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        robot_frames.append(frame[:, int(frame.shape[1] / 4.2) : -int(frame.shape[1] / 4.2)])
    robot_frames = [robot_frames[i] for i in range(0, len(robot_frames), 5)]

    cap = cv2.VideoCapture(human_video_path)
    human_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = frame[int(frame.shape[0] / 3.5) : int(1.7 * frame.shape[0] / 3), :]
        frame = frame[: -int(frame.shape[0] / 2), :]
        human_frames.append(cv2.resize(frame, (int(frame.shape[1] / 2.2), int(frame.shape[0] / 2.2))))

    h = robot_frames[0].shape[0]
    w = robot_frames[0].shape[1]
    p = len(human_frames) / len(robot_frames)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(len(robot_frames)):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, : robot_frames[0].shape[1]] = robot_frames[i]
        frame[-human_frames[0].shape[0] :, -human_frames[0].shape[1] :] = human_frames[int(i * p)]
        out.write(frame)
        cv2.imwrite("frame.png", frame)
    out.release()


if __name__ == "__main__":
    args = get_args()

    robot_video_path = f"data/motion/{args.motion_id}/{args.motion_id}_robot.mp4"
    human_video_path = f"data/motion/{args.motion_id}/{args.motion_id}_human.mp4"
    output_path = f"data/motion/{args.motion_id}/{args.motion_id}_stitched.mp4"
    stitch_videos(robot_video_path, human_video_path, output_path)
