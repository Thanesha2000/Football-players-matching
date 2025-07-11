# Soccer Player Re-Identification: Cross-Camera Player Mapping

This project implements a Cross-Camera Player Mapping system for a Soccer Player Re-Identification assignment. It utilizes the **Ultralytics YOLOv11** model for player detection and a simple feature extraction and matching pipeline to perform re-identification of players across two camera feeds.

## 📌 Project Overview

- **Goal**: Match soccer players between two video feeds from different camera angles.
- **Input**: Two video files (e.g., `broadcast.mp4` and `tacticam.mp4`).
- **Output**: Detected players with matched IDs across both views.
- **Detection Model**: YOLOv11 (Ultralytics)
- **Tools Used**: Python, OpenCV, NumPy, Git LFS (for large files)

## 🛠️ Features

- Player detection using YOLOv11
- Frame-by-frame player cropping
- Feature extraction using color histograms or embeddings
- Matching based on feature similarity
- Visualization of matched players across cameras
