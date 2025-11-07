import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import cv2

def compute_video_stats(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        return None

    frames_np = np.stack(frames)  # (N, H, W, 3)
    return frames_np

def get_stats(base_dir,camera):
    video_paths = glob(os.path.join(base_dir, f"chunk-*/observation.images.{camera}/*.mp4"))

    stats = {
        "sum": np.zeros(3, dtype=np.float64),
        "sum_squared": np.zeros(3, dtype=np.float64),
        "min": np.full(3, np.inf),
        "max": np.full(3, -np.inf),
        "total_pixels": 0
    }

    for path in tqdm(video_paths, desc=f"Processing {camera}"):
        frames = compute_video_stats(path)
        if frames is None:
            continue

        frames = frames.astype(np.float32)  # 정규화 (0~1)
        reshaped = frames.reshape(-1, 3)  # (N * H * W, 3)

        stats["sum"] += reshaped.sum(axis=0)
        stats["sum_squared"] += (reshaped ** 2).sum(axis=0)
        stats["min"] = np.minimum(stats["min"], reshaped.min(axis=0))
        stats["max"] = np.maximum(stats["max"], reshaped.max(axis=0))
        stats["total_pixels"] += reshaped.shape[0]

    if stats["total_pixels"] == 0:
        return None

    mean = stats["sum"] / stats["total_pixels"]
    std = np.sqrt(stats["sum_squared"] / stats["total_pixels"] - mean ** 2)

    return {
        "camera": camera,
        "mean": mean,
        "std": std,
        "min": stats["min"],
        "max": stats["max"]
    }


if __name__=='__main__':
    base_dir = "/data/piper_grape0711_10/videos"
    cameras = ["table", "wrist"]

    for camera in cameras:
        stats = get_stats(base_dir, camera)
        print(stats)