import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ground truth measurements
GROUND_TRUTH = {
    "shoulder_to_elbow": 28.0,  # cm
    "elbow_to_wrist": 27.0  ,    # cm
    "shoulder_width": 38.0     # cm
}
import numpy as np

# === Load CSVs ===
gt_file = "ground_truth_distances.csv"
yolo_file = "output/yolo_distances.csv"
mp_file = "output/mediapipe_distances.csv"

gt_df = pd.read_csv(gt_file)
yolo_df = pd.read_csv(yolo_file)
mp_df = pd.read_csv(mp_file)

# Desired frames
desired_frames = [0, 50]

# Helper: find nearest available frame in a dataframe
def nearest_frame(df, target):
    frames = np.sort(df['frame'].unique())
    if target in frames:
        return target
    # find nearest
    idx = np.argmin(np.abs(frames - target))
    return int(frames[idx])

# Resolve frames present in ground truth (prefer using ground truth frames if available)
available_frames = sorted(gt_df['frame'].unique())
frames_to_plot = []
for f in desired_frames:
    frames_to_plot.append(nearest_frame(gt_df, f) if len(available_frames) > 0 else f)

# Ensure the other dataframes contain the chosen frames; if not, we'll pick nearest in them too when reading rows

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

# Helper to build ground truth row according to rules:
# - shoulder width uses GROUND_TRUTH['shoulder_width'] mapped to left_shoulder_to_right_shoulder_cm
# - shoulder_to_elbow and elbow_to_wrist use GROUND_TRUTH values for both left and right shoulder-elbow and elbow-wrist
# - other columns (hips, movements) come from gt_df if available for that frame
def build_gt_row(frame):
    row = {}
    # try to get the calculated ground truth row if present
    gt_frame_row = gt_df[gt_df['frame'] == frame]
    for c in cols:
        row[c] = None

    # shoulder width -> map into 'shoulder_width' column
    if 'left_shoulder_to_right_shoulder_cm' in gt_df.columns and not gt_frame_row.empty and not pd.isna(gt_frame_row['left_shoulder_to_right_shoulder_cm'].values[0]):
        row['shoulder_width'] = float(gt_frame_row['left_shoulder_to_right_shoulder_cm'].values[0])
    else:
        row['shoulder_width'] = GROUND_TRUTH['shoulder_width']

    # shoulder->elbow and elbow->wrist map
    for side in ['left', 'right']:
        se_col = f"{side}_shoulder_to_{side}_elbow_cm"
        ew_col = f"{side}_elbow_to_{side}_wrist_cm"
        # prefer calculated gt if present
        if se_col in gt_df.columns and not gt_frame_row.empty and not pd.isna(gt_frame_row[se_col].values[0]):
            se_val = float(gt_frame_row[se_col].values[0])
        else:
            se_val = GROUND_TRUTH['shoulder_to_elbow']
        if ew_col in gt_df.columns and not gt_frame_row.empty and not pd.isna(gt_frame_row[ew_col].values[0]):
            ew_val = float(gt_frame_row[ew_col].values[0])
        else:
            ew_val = GROUND_TRUTH['elbow_to_wrist']
        # map into expected cols if present
        if f"{side}_shoulder_to_{side}_elbow_cm" in cols:
            row[f"{side}_shoulder_to_{side}_elbow_cm"] = se_val
        # Note: elbow->wrist columns are not requested as separate cols in this plot; movements are used instead

    # For the rest, try to pick from gt_frame_row if available
    for c in cols:
        if c in gt_df.columns and not gt_frame_row.empty and not pd.isna(gt_frame_row[c].values[0]):
            row[c] = float(gt_frame_row[c].values[0])

    # ensure all keys exist in the same order as cols
    return [row[c] if row[c] is not None else np.nan for c in cols]


# === Plotting ===
for frame in frames_to_plot:
    # find nearest frame in each df if exact frame missing
    frame_yolo = nearest_frame(yolo_df, frame) if 'frame' in yolo_df.columns else frame
    frame_mp = nearest_frame(mp_df, frame) if 'frame' in mp_df.columns else frame

    gt_row = build_gt_row(frame)

    # Read rows from YOLO and MP (if missing, fill nan)
    def read_row(df, frame_key):
        if frame_key in df['frame'].values:
            r = df[df['frame'] == frame_key]
            return [float(r[c].values[0]) if c in r.columns and not pd.isna(r[c].values[0]) else np.nan for c in cols]
        else:
            return [np.nan] * len(cols)

    yolo_row = read_row(yolo_df, frame_yolo)
    mp_row = read_row(mp_df, frame_mp)

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
    out_path = f"output/comparison_frame_{frame}.png"
    plt.savefig(out_path)
    print(f"Saved comparison plot: {out_path}")
    plt.show()
