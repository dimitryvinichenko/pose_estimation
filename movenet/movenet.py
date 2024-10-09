import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

import time


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)



cap = cv2.VideoCapture('test1.mp4')
# video duration
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps


video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('output', fourcc, fps, (video_width, video_height))



#load model
model = 'thunder'

if model == 'thunder'
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")  
    input_size = 256 #image size for movenet thunder model
else: 
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192 #image size for movenet thunder model


model = module.signatures['serving_default'] #regime for inferrence

start = time.time()

while cap.isOpened():

    ret, frame = cap.read()
    try:    
        # Reshape image
        input_image = frame.copy()
        input_image = tf.image.resize_with_pad(np.expand_dims(input_image, axis=0), 
                                            input_size, input_size)
        input_image = tf.cast(input_image, dtype=tf.int32)


        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()

        print(keypoints_with_scores)

        draw_keypoints(frame, keypoints_with_scores, 0.4)
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        
        cv2.imshow('MoveNet Lightning', frame)
    except:
        break

end = time.time()
    
cap.release()
cv2.destroyAllWindows()

print(f'Video duration: {duration}')
print(f'Video processing: {end - start}')

