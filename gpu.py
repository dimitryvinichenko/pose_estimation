import sys
import os
import cv2
import tqdm

import mediapipe as mp
import mediapipe.python.solutions.pose as sp
import mediapipe.python.solutions.drawing_utils as du

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode

from urllib.request import urlretrieve



def convert_landmarks(detection_result):
    normalized = landmark_pb2.NormalizedLandmarkList()
    for v in detection_result.pose_landmarks:
        normalized.landmark.extend([landmark_pb2.NormalizedLandmark(x=u.x, y=u.y, z=u.z) for u in v])

    return normalized


def ensure_file(file, url):
    if os.path.isfile(file):
        return
    urlretrieve(url, file)


def draw_pose_to_video(input_file="input.mp4", output_file="output.mp4"):

    task_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    model_file = 'pose_landmarker_heavy.task'

    ensure_file(file=model_file, url=task_url)

    options = BaseOptions(model_asset_path=model_file, delegate=BaseOptions.Delegate.GPU)
    options = PoseLandmarkerOptions(base_options=options, running_mode=RunningMode.VIDEO)

    with PoseLandmarker.create_from_options(options) as landmarker:
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
        poses = [None]*frame_count
        for i in tqdm.tqdm(range(frame_count)):
            ok, img = cap.read()
            if not ok:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=img)
            ts = int(i * 1000.0 / fps)
            poses[i] = landmarker.detect_for_video(mp_image, ts)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(frame_count):
            ok, img = cap.read()
            if not ok:
                continue

            pose = poses[i]
            if pose is not None:
                du.draw_landmarks(image=img,
                                  landmark_list=convert_landmarks(pose),
                                  connections=sp.POSE_CONNECTIONS)
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
