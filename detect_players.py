from ultralytics import YOLO
import cv2
import os
import json

# Define paths
input_dir = './frames/'
output_dir = './frames_detected/'
model_path = r'C:\Users\thane\OneDrive\Desktop\anubhav\models\best.pt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Dictionary to store detections
detections = {}

for frame_file in os.listdir(input_dir):
    if frame_file.endswith('.jpg'):
        frame_path = os.path.join(input_dir, frame_file)
        print(f"Processing {frame_file}")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Could not load {frame_file}")
            continue
        # Resize to 640x384
        frame = cv2.resize(frame, (640, 384))
        height, width, _ = frame.shape
        print(f"{frame_file} resized to: {width}x{height}")

        results = model(frame)
        frame_boxes = []
        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = results[0].boxes.conf[i]
            if confidence > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Player {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame_boxes.append([x1, y1, x2, y2])

        output_path = os.path.join(output_dir, frame_file.replace('.jpg', '_detected.jpg'))
        cv2.imwrite(output_path, frame)
        print(f"Saved detected frame to {output_path}")
        detections[frame_file] = frame_boxes

# Save detections to JSON
with open('detections.json', 'w') as f:
    json.dump(detections, f)
print("Player detection complete!")