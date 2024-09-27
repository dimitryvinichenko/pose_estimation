import cv2
import time
import poseModule as pm
import os  
import pandas as pd

# from parser import parser
import argparse

args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default='input.mp4')
args.add_argument('--output', type=str, default='output.mp4')

args = args.parse_args()

def main():
    video = str(args.input)
    cap = cv2.VideoCapture(video)

    # video duration
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    pTime = time.time()
    detector = pm.poseDetector()
    start = pTime

    # Get actual frame width and height from the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    for frame in range(frame_count):
        
        _ , img = cap.read()
        img = detector.drawPose(img)
        # lmList = detector.findPose(img, draw=False)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3) 

        cv2.imshow("Image", img)

        out.write(img)

        cv2.waitKey(1)
    
    end = time.time()
    processing_time = end - start

    df = pd.DataFrame({'Video Duration': [duration], 'Processing time': [processing_time]})        
    df.to_csv('report.csv', index=False)

    out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()