import cv2
import os
import numpy as np
import json

input_dir = './frames_detected/'
output_dir = './frames_aligned/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

src_points = np.load('src_points.npy').astype(np.float32)
dst_points = np.load('dst_points.npy').astype(np.float32)

h, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

with open('detections.json', 'r') as f:
    detections = json.load(f)

for frame_file in os.listdir(input_dir):
    if frame_file.endswith('_detected.jpg'):
        frame_path = os.path.join(input_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Could not load {frame_file}")
            continue
        base_file = frame_file.replace('_detected.jpg', '.jpg')
        boxes = detections.get(base_file, [])
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            box_np = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_box = cv2.perspectiveTransform(box_np, h)
            if transformed_box.shape[0] >= 2:
                x1_t, y1_t = map(int, transformed_box[0, 0])
                x2_t, y2_t = map(int, transformed_box[1, 0])
                # Clip to 640x384 bounds
                x1_t = max(0, min(639, x1_t))
                y1_t = max(0, min(383, y1_t))
                x2_t = max(0, min(639, x2_t))
                y2_t = max(0, min(383, y2_t))
                cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), (255, 0, 0), 2)

        output_path = os.path.join(output_dir, frame_file.replace('_detected', '_aligned'))
        cv2.imwrite(output_path, frame)
        print(f"Saved aligned frame to {output_path}")

print("Camera alignment complete!")