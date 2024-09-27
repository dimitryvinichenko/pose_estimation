import cv2
import time
import poseModule as pm
import os  
import pandas as pd


def main():
    video = str('IMG_2332.MP4')
    cap = cv2.VideoCapture(video)

    # video duration
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    pTime = time.time()
    detector = pm.poseDetector()
    start = pTime

    for frame in range(frame_count):
        
        _ , img = cap.read()
        img = detector.drawPose(img)
        lmList = detector.findPose(img, draw=False)
        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
    end = time.time()
    processing_time = end - start

    df = pd.DataFrame({'Video Duration': [duration], 'Processing time': [processing_time]})        
    df.to_csv('report.csv', index=False)


if __name__ == "__main__":
    main()