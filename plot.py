import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Ground Truth reference values (static) ===
GROUND_TRUTH = {
    "shoulder_to_elbow": 28.0,  # cm
    "elbow_to_wrist": 27.0,     # cm
    "shoulder_width": 38.0      # cm
}

# === Load CSVs ===
gt_file = "ground_truth_distances.csv"
yolo_file = "output/yolo_distances.csv"
mp_file = "output/mediapipe_distances.csv"

# Load into DataFrames
gt_df = pd.read_csv(gt_file)
yolo_df = pd.read_csv(yolo_file)
mp_df = pd.read_csv(mp_file)

# === Columns to compare ===
cols = [
    "shoulder_width",
    "left_shoulder_to_left_elbow_cm",
    "right_shoulder_to_right_elbow_cm",
    "left_wrist_movement_cm",
    "right_wrist_movement_cm",
    "left_elbow_movement_cm",
    "right_elbow_movement_cm"
]

# === Prepare Ground Truth Data ===
# For ground truth, we’ll use fixed reference values unless they’re present in gt_df.
gt_df_filled = pd.DataFrame()
gt_df_filled['frame'] = gt_df['frame'] if 'frame' in gt_df.columns else yolo_df['frame']

for col in cols:
    if col in gt_df.columns:
        gt_df_filled[col] = gt_df[col]
    else:
        # Fill with appropriate ground truth constant
        if 'shoulder_to_elbow' in col:
            gt_df_filled[col] = GROUND_TRUTH['shoulder_to_elbow']
        elif 'elbow_to_wrist' in col:
            gt_df_filled[col] = GROUND_TRUTH['elbow_to_wrist']
        elif 'shoulder_width' in col:
            gt_df_filled[col] = GROUND_TRUTH['shoulder_width']
        else:
            gt_df_filled[col] = np.nan

# === Align frames across all datasets ===
common_frames = sorted(
    set(yolo_df['frame']).union(set(mp_df['frame'])).union(set(gt_df_filled['frame']))
)

yolo_df = yolo_df.set_index('frame').reindex(common_frames).reset_index()
mp_df = mp_df.set_index('frame').reindex(common_frames).reset_index()
gt_df_filled = gt_df_filled.set_index('frame').reindex(common_frames).reset_index()

# === Plotting loop for each column ===
for col in cols:
    plt.figure(figsize=(10, 5))

    # Plot each source as a line
    plt.plot(yolo_df['frame'], yolo_df[col], label='YOLO', color='orange', linewidth=2, marker='o')
    plt.plot(mp_df['frame'], mp_df[col], label='Mediapipe', color='green', linewidth=2, marker='x')
    plt.plot(gt_df_filled['frame'], gt_df_filled[col], label='Ground Truth', color='blue', linewidth=2, linestyle='--')

    plt.title(f"{col.replace('_', ' ').title()} Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Distance (cm)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the figure
    out_path = f"output/linegraph_{col}.png"
    plt.savefig(out_path)
    print(f"✅ Saved: {out_path}")
    plt.show()
