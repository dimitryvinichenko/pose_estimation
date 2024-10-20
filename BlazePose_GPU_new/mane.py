import mediapipe as mp
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
import argparse


args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default='input.mp4')
args.add_argument('--output', type=str, default='output.mp4')

args = args.parse_args()


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='pose_landmarker.task',
        delegate=BaseOptions.Delegate.GPU
    ),
    running_mode=VisionRunningMode.IMAGE, 
    # min_pose_detection_confidence=0.9,
    min_tracking_confidence=0.9
)


with PoseLandmarker.create_from_options(options) as landmarker:

    video = str(args.input)
    cap = cv2.VideoCapture(video)

    # video duration
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    # Get actual frame width and height from the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    start = time.time()

    results = []

    for frame in range(frame_count):
        _, image = cap.read()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = landmarker.detect(mp_image)
        results.append(result)

    print(f'Duration: {duration}')
    print(f'Processing time: {time.time() - start}')

    cap = cv2.VideoCapture(video)

    for frame in range(frame_count):
        
        _, image = cap.read()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), results[frame])
        out.write(annotated_image)



    cap.release()
    cv2.destroyAllWindows()


