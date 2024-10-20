# pose_estimation

```bash
conda env create -n pose_estimation -f environment.yml
```
```bash
conda activate pose_estimation
```

To load model
```bash
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

```bash
python3 mane.py --input some_input_video.mp4 --output some_output_video.mp4 
```



