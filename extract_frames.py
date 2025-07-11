import cv2
import os

# Define paths and video filenames
input_dir = r'C:\Users\thane\OneDrive\Desktop\anubhav\videos'  # Raw string for Windows path
output_dir = './frames/'
videos = ['broadcast.mp4', 'tacticam.mp4']

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each video
for video_name in videos:
    video_path = os.path.join(input_dir, video_name)
    print(f"Trying to open: {video_path}")  # Debug print to check path
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist.")
        continue
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        continue

    frame_count = 0
    skip_frame = 10  # Extract every 10th frame (adjust as needed)
    skip_count = 0

    # Get video properties for synchronization
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_name}: FPS={fps}, Total Frames={total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if skip_count % skip_frame == 0:
            # Save frame with unique name
            frame_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        skip_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_name}")

# Simple synchronization check (manual alignment example)
sync_event_frame_broadcast = 100  # Example: Goal at frame 100 in broadcast.mp4
sync_event_frame_tacticam = 120  # Example: Goal at frame 120 in tacticam.mp4
offset = sync_event_frame_tacticam - sync_event_frame_broadcast
print(f"Synchronization offset: {offset} frames")

print("Frame extraction and synchronization check complete!")