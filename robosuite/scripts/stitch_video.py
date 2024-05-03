import argparse
import re

import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_id", type=str, default="000003")
    args = parser.parse_args()
    return args


def stitch_videos_human_robot(robot_video_path, human_video_path, output_path):
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


def stitch_2_2_videos(video_paths):
    video_clips = [cv2.VideoCapture(path) for path in video_paths]
    width = int(video_clips[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_clips[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(f"tmp_video/stitched_{video_idx}.mp4", fourcc, 3, (width * 2, height))
    out = cv2.VideoWriter(f"tmp_video/stitched_adjust.mp4", fourcc, 3, (width * 2, height * 2))

    data = [
        [0, 0, 0, 0, 1, 0.0318287],
        [1, 0, 1, 0, 1, 0.0189972],
        [0, 0, 0, 0, 1, 0.0413229],
        [1, 0, 1, 0, 1, 0.0603404],
    ]
    cap = [
        "interpolate 50 + adjust 20",
        "interpolate 50 + adjust 20",
        "interpolate 70 + adjust 0",
        "interpolate 70 + adjust 0",
    ]

    out_frames = []
    while True:
        frames = [cap.read()[1] for cap in video_clips]  # read a frame from each video
        if any(frame is None for frame in frames):
            break  # stop if any video reaches the end

        # add text on each frame
        for i, frame in enumerate(frames):
            w_offset = 50
            h_offset = 50
            cv2.putText(
                frame,
                f"shoulder",
                (10 + w_offset, 50 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][0] > 0.5)),
                2,
            )  # {data[-1][i]:.2f}
            cv2.putText(
                frame,
                f"elbow",
                (10 + w_offset, 100 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][2] > 0.5)),
                2,
            )
            cv2.putText(
                frame,
                f"wrist",
                (10 + w_offset, 150 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][4] > 0.5)),
                2,
            )

            cv2.putText(frame, f"w", (170 + w_offset, 20 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(
                frame,
                f"{data[i][0]:.1f}",
                (150 + w_offset, 50 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][0] > 0.5)),
                2,
            )
            cv2.putText(
                frame,
                f"{data[i][2]:.1f}",
                (150 + w_offset, 100 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][2] > 0.5)),
                2,
            )
            cv2.putText(
                frame,
                f"{data[i][4]:.1f}",
                (150 + w_offset, 150 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][4] > 0.5)),
                2,
            )

            cv2.putText(frame, f"err", (270 + w_offset, 20 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(
                frame,
                f"N/A",
                (250 + w_offset, 50 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][0] > 0.5)),
                2,
            )
            cv2.putText(
                frame,
                f"N/A",
                (250 + w_offset, 100 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][2] > 0.5)),
                2,
            )
            cv2.putText(
                frame,
                f"{data[i][5]:.4f}",
                (250 + w_offset, 150 + h_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 225 * int(data[i][4] > 0.5)),
                2,
            )

            cv2.putText(frame, f"{cap[i]}", (200, 900), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        # Assume we're doing a 2x4 grid as rows of 4 videos each
        row1 = np.hstack((frames[0], frames[1]))
        row2 = np.hstack((frames[2], frames[3]))

        # Vertically stack the rows
        final_frame = np.vstack((row1, row2))

        # Write the stitched frame
        out.write(final_frame)
        out_frames.append(final_frame)

    # Release everything when done
    for cap in video_clips:
        cap.release()
    out.release()


def stitch_3_3_videos(video_paths):
    video_clips = [cv2.VideoCapture(path) for path in video_paths]
    width = int(video_clips[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_clips[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"tmp_video/stitched_reach.mp4", fourcc, 20, (int(512 * 3), int(512 * 3)))

    out_frames = []
    while True:
        frames = [cap.read()[1] for cap in video_clips]  # read a frame from each video
        if any(frame is None for frame in frames):
            break  # stop if any video reaches the end

        new_frames = []
        for i, frame in enumerate(frames):
            frame = cv2.resize(frame, (512, 512))
            # video_name = video_paths[i][10:-4]
            # # extract three numbers from the video name
            # offsets = re.findall(r"[-+]?\d*\.\d+|\d+", video_name)
            # assert(len(offsets) == 3)

            # # print(video_name)
            # cv2.putText(frame, f'offset=[{offsets[0]}, {offsets[1]}, {offsets[2]}]', (90, 290), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 2)
            new_frames.append(frame)

        # Assume we're doing a 3x3 grid as rows of 3 videos each
        row1 = np.hstack((new_frames[0], new_frames[1], new_frames[2]))
        row2 = np.hstack((new_frames[3], new_frames[4], new_frames[5]))
        row3 = np.hstack((new_frames[6], new_frames[7], new_frames[8]))

        # Vertically stack the rows
        final_frame = np.vstack((row1, row2, row3))

        # Write the stitched frame
        out.write(final_frame)
        out_frames.append(final_frame)

    # Release everything when done
    for cap in video_clips:
        cap.release()
    out.release()


if __name__ == "__main__":
    args = get_args()

    robot_video_path = f"data/motion/{args.motion_id}/{args.motion_id}_robot.mp4"
    human_video_path = f"data/motion/{args.motion_id}/{args.motion_id}_human.mp4"
    output_path = f"data/motion/{args.motion_id}/{args.motion_id}_stitched.mp4"
    # stitch_videos_human_robot(robot_video_path, human_video_path, output_path)

    # video_paths = [
    #     'tmp_video/0.1_50_20_adjust_0.mp4',
    #     'tmp_video/0.1_50_20_adjust_1.mp4',
    #     'tmp_video/0.1_70_0_adjust_0.mp4',
    #     'tmp_video/0.1_70_0_adjust_1.mp4',
    # ]
    # stitch_2_2_videos(video_paths)

    # video_paths = [
    #     'tmp_video/0x0y0z.mp4',
    #     'tmp_video/0x0y0.05z.mp4',
    #     'tmp_video/0x0y0.1z.mp4',
    #     'tmp_video/0x0.05y0z.mp4',
    #     'tmp_video/0x0.1y0z.mp4',
    #     'tmp_video/0.05x0y0z.mp4',
    #     'tmp_video/0.05x0.05y0z.mp4',
    #     'tmp_video/0.1x0.1y0z.mp4',
    #     'tmp_video/0.1x0.2y0.05z.mp4',
    # ]
    # stitch_3_3_videos(video_paths)

    video_paths = [
        "tmp_video/0.mp4",
        "tmp_video/1.mp4",
        "tmp_video/2.mp4",
        "tmp_video/3.mp4",
        "tmp_video/4.mp4",
        "tmp_video/5.mp4",
        "tmp_video/6.mp4",
        "tmp_video/7.mp4",
        "tmp_video/8.mp4",
    ]
    stitch_3_3_videos(video_paths)
