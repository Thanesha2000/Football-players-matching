import cv2
import os
import numpy as np
import json

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_t, y1_t, x2_t, y2_t = box2
    inter_x1 = max(x1, x1_t)
    inter_y1 = max(y1, y1_t)
    inter_x2 = min(x2, x2_t)
    inter_y2 = min(y2, y2_t)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_t - x1_t) * (y2_t - y1_t)
    return inter_area / float(area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

src_points = np.load('src_points.npy').astype(np.float32)
dst_points = np.load('dst_points.npy').astype(np.float32)
h, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

with open('detections.json', 'r') as f:
    detections = json.load(f)

for broadcast_file in detections.keys():
    if 'broadcast' in broadcast_file:
        tacticam_file = broadcast_file.replace('broadcast', 'tacticam')
        if tacticam_file in detections:
            broadcast_boxes = detections[broadcast_file]
            tacticam_boxes = detections[tacticam_file]

            for b_box in broadcast_boxes:
                b_box_np = np.array([[b_box[0], b_box[1]], [b_box[2], b_box[3]]], dtype=np.float32).reshape(-1, 1, 2)
                transformed_box = cv2.perspectiveTransform(b_box_np, h)
                if transformed_box.shape[0] >= 2:
                    x1_t, y1_t = map(int, transformed_box[0, 0])
                    x2_t, y2_t = map(int, transformed_box[1, 0])
                    print(f"Debug: Transformed box for {broadcast_file}: ({x1_t}, {y1_t}, {x2_t}, {y2_t})")

                    best_iou = 0
                    best_match = None
                    for t_box in tacticam_boxes:
                        x1_t2, y1_t2, x2_t2, y2_t2 = map(int, t_box)
                        iou = calculate_iou((x1_t, y1_t, x2_t, y2_t), (x1_t2, y1_t2, x2_t2, y2_t2))
                        if iou > best_iou:
                            best_iou = iou
                            best_match = t_box
                    print(f"Checking {broadcast_file}: Transformed ({x1_t, y1_t, x2_t, y2_t}), IoU={best_iou}")
                    if best_iou > 0.1:  # Temporary lower threshold
                        print(f"Match found for {broadcast_file}: {b_box} -> {best_match}")

print("Player matching complete!")