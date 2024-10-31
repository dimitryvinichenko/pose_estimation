import subprocess
import sys
import traceback

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:

    import cv2
    from tqdm import tqdm
    import mediapipe.python.solutions.pose as mp
    import mediapipe.python.solutions.drawing_utils as du

except ImportError:
    install('tqdm')
    from tqdm import tqdm

    install('mediapipe')
    import mediapipe.python.solutions.pose as mp
    import mediapipe.python.solutions.drawing_utils as du

    install('opencv-contrib-python')
    import cv2


def process_video_cpu(input_file: str, output_file: str) -> None:
    """Рисует позу человека на кажом кадре (CPU)"""
    cap = cv2.VideoCapture()
    ok = cap.open(input_file)

    assert ok, "can`t open file"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename=output_file,
                             fourcc=fourcc,
                             fps=fps,
                             frameSize=(frame_width, frame_height))

    detector = mp.Pose(enable_segmentation=False,
                       model_complexity=0)

    poses = [None]*frame_count
    for i in tqdm(range(frame_count), desc="Detect"):
        ok, img = cap.read()
        if not ok:
            continue

        poses[i] = detector.process(img)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in tqdm(range(frame_count), desc="Draw"):
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
    try:
        process_video_cpu(sys.argv[1], "bar.mp4")
    except Exception as e:
        print(traceback.format_exc())