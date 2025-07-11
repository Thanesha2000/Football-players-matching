import numpy as np

# Load points
src_points = np.load('src_points_auto.npy')
dst_points = np.load('dst_points_auto.npy')

print("Source Points (Broadcast):")
print(src_points)
print("Destination Points (Tacticam):")
print(dst_points)