import cv2
import numpy as np

# Load images
broadcast_frame = cv2.imread('C:\\Users\\thane\\OneDrive\\Desktop\\anubhav\\frames_detected\\broadcast_frame_000012_detected.jpg', 0)
tacticam_frame = cv2.imread('C:\\Users\\thane\\OneDrive\\Desktop\\anubhav\\frames_detected\\tacticam_frame_000012_detected.jpg', 0)

if broadcast_frame is None or tacticam_frame is None:
    print("Error: One or both images not found. Check file paths.")
    exit()

# Apply Gaussian blur
broadcast_frame = cv2.GaussianBlur(broadcast_frame, (5, 5), 0)
tacticam_frame = cv2.GaussianBlur(tacticam_frame, (5, 5), 0)

# Initialize SIFT
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(broadcast_frame, None)
kp2, des2 = sift.detectAndCompute(tacticam_frame, None)

# BFMatcher with ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Filter good matches and remove duplicates
good_matches = []
seen_pairs = set()
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        query_idx = m.queryIdx
        train_idx = m.trainIdx
        pair = (query_idx, train_idx)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            good_matches.append(m)

# Extract matched points
src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# Ensure equal number of points (6-8)
num_points = min(8, max(6, min(len(src_points), len(dst_points))))
if num_points < 6:
    print("Warning: Insufficient points. Using available points.")
src_points = src_points[:num_points]
dst_points = dst_points[:num_points]

# Save points
np.save('src_points_auto.npy', src_points)
np.save('dst_points_auto.npy', dst_points)

# Print points
print("Source Points (Broadcast):")
for i, point in enumerate(src_points):
    print(point)
print("Destination Points (Tacticam):")
for i, point in enumerate(dst_points):
    print(point)
print(f"Saved {num_points} auto src_points and {num_points} auto dst_points.")