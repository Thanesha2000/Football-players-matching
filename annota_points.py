import cv2
import numpy as np

# Paths to the frames
broadcast_frame = r'C:\Users\thane\OneDrive\Desktop\anubhav\frames_detected\broadcast_frame_000006_detected.jpg'
tacticam_frame = r'C:\Users\thane\OneDrive\Desktop\anubhav\frames_detected\tacticam_frame_000012_detected.jpg'

# Lists to store points
src_points = []
dst_points = []

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red dot
        cv2.imshow('image', img)
        print(f"Point added: ({x}, {y})")

# Process broadcast frame
img = cv2.imread(broadcast_frame)
cv2.imshow('image', img)
cv2.setMouseCallback('image', mouse_callback, src_points)
print("Click 4 or more points on the broadcast frame (e.g., field corners). Press 'q' to finish.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# Process tacticam frame
img = cv2.imread(tacticam_frame)
cv2.imshow('image', img)
cv2.setMouseCallback('image', mouse_callback, dst_points)
print("Click 4 or more corresponding points on the tacticam frame. Press 'q' to finish.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# Save points to file for use in align_cameras.py
np.save('src_points.npy', np.array(src_points))
np.save('dst_points.npy', np.array(dst_points))
print(f"Saved {len(src_points)} src_points and {len(dst_points)} dst_points.")