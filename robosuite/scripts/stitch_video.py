import cv2
import numpy as np

if __name__ == "__main__":
    # read in reference video as a list of images
    # read mp4 video
    cap = cv2.VideoCapture("robot2.mp4")
    robot_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        robot_frames.append(frame[:, int(frame.shape[1] / 4) :])
    robot_frames = [robot_frames[i] for i in range(0, len(robot_frames), 5)]
    print(len(robot_frames), robot_frames[0].shape)  # 718, 1278, 3

    cap = cv2.VideoCapture("human.mp4")
    human_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[int(frame.shape[0] / 3.5) : int(1.7 * frame.shape[0] / 3), :]
        human_frames.append(cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2))))
    print(len(human_frames), human_frames[0].shape)  # 1280, 720, 3

    # stitch the two videos together
    # create a VideoWriter object
    h = robot_frames[0].shape[0]
    w = robot_frames[0].shape[1]
    p = len(human_frames) / len(robot_frames)

    out = cv2.VideoWriter("track_hand.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(len(robot_frames)):
        # write the frame
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, : robot_frames[0].shape[1]] = robot_frames[i]
        frame[-human_frames[0].shape[0] :, -human_frames[0].shape[1] :] = human_frames[int(i * p)]
        out.write(frame)
        cv2.imwrite("frame.png", frame)
    out.release()
