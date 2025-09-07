import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load CSVs ===
gt_file = "ground_truth_distances.csv"
yolo_file = "output/yolo_distances.csv"
mp_file = "output/mediapipe_distances.csv"

gt_df = pd.read_csv(gt_file)
yolo_df = pd.read_csv(yolo_file)
mp_df = pd.read_csv(mp_file)

# === Select frames to compare ===
frames_to_plot = [gt_df['frame'].iloc[0], gt_df['frame'].iloc[-1]]  # first and last frame in ground truth
# Make sure these frames exist in all CSVs
gt_df = gt_df[gt_df['frame'].isin(frames_to_plot)]
yolo_df = yolo_df[yolo_df['frame'].isin(frames_to_plot)]
mp_df = mp_df[mp_df['frame'].isin(frames_to_plot)]

# === Columns to compare ===
cols = [
    "left_shoulder_to_right_shoulder_cm",
    "left_hip_to_right_hip_cm",
    "left_shoulder_to_left_elbow_cm",
    "right_shoulder_to_right_elbow_cm",
    "left_shoulder_to_left_hip_cm",
    "right_shoulder_to_right_hip_cm",
    "left_wrist_movement_cm",
    "right_wrist_movement_cm",
    "left_elbow_movement_cm",
    "right_elbow_movement_cm"
]

# === Plotting ===
for frame in frames_to_plot:
    gt_row = gt_df[gt_df['frame'] == frame][cols].values[0]
    yolo_row = yolo_df[yolo_df['frame'] == frame][cols].values[0]
    mp_row = mp_df[mp_df['frame'] == frame][cols].values[0]

    x = np.arange(len(cols))  # positions for bars
    width = 0.25  # bar width

    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(x - width, gt_row, width, label='Ground Truth', color='blue')
    ax.bar(x, yolo_row, width, label='YOLO', color='orange')
    ax.bar(x + width, mp_row, width, label='Mediapipe', color='green')

    ax.set_ylabel('Distance (cm)')
    ax.set_title(f'Frame {frame} - Ground Truth vs YOLO vs Mediapipe')
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()
