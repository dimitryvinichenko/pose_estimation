import sys
import tqdm
import cv2
import mediapipe.python.solutions.pose as mp
import mediapipe.python.solutions.drawing_utils as du


def draw_pose_to_video(input_file="input.mp4", output_file="output.mp4"):
    cap = cv2.VideoCapture()
    assert cap.open(input_file), "can`t open file"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename=output_file,
                             fourcc=fourcc,
                             fps=fps,
                             frameSize=(frame_width, frame_height))

    detector = mp.Pose(enable_segmentation=False, model_complexity=0)

    poses = [None]*frame_count
    for i in tqdm.tqdm(range(frame_count)):
        ok, img = cap.read()
        if not ok:
            continue

        poses[i] = detector.process(img)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(frame_count):
        ok, img = cap.read()
        if not ok:
            continue

        pose = poses[i]
        if pose is not None:
            du.draw_landmarks(image=img,
                              landmark_list=pose.pose_landmarks,
                              connections=mp.POSE_CONNECTIONS)
        writer.write(img)

    cap.release()
    writer.release()


if __name__ == "__main__":
    input_file = "input.mp4"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    output_file = "output.mp4"
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    draw_pose_to_video(input_file, output_file)
