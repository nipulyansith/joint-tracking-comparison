import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === File paths ===
GT_FILE = "ground_truth_distances.csv"
MP_FILE = "output/mediapipe_distances.csv"
YOLO_FILE = "output/yolo_distances.csv"
OUT_DIR = "plottedImages"

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# === Load all datasets ===
gt = pd.read_csv(GT_FILE)
mp = pd.read_csv(MP_FILE)
yolo = pd.read_csv(YOLO_FILE)

# === Columns to compare ===
# Include both distances (_cm) and angles (_angle)
columns_to_plot = [
    # Movement distances
    "left_wrist_movement_cm", "right_wrist_movement_cm",
    "left_elbow_movement_cm", "right_elbow_movement_cm",
    "left_shoulder_movement_cm", "right_shoulder_movement_cm",
    "left_knee_movement_cm", "right_knee_movement_cm",

    # Example angles (ensure they exist in your CSVs)
    "left_elbow_angle_deg", "right_elbow_angle_deg"
]

# === Prepare results list ===
results = []

# === Loop through each selected column ===
for col in columns_to_plot:
    if col not in gt.columns:
        print(f"⚠️ Column '{col}' not found in Ground Truth file. Skipping...")
        continue

    if col not in mp.columns or col not in yolo.columns:
        print(f"⚠️ Column '{col}' not found in Mediapipe or YOLO file. Skipping...")
        continue

    # Align frames
    merged = gt[["frame", col]].merge(mp[["frame", col]], on="frame", suffixes=("_gt", "_mp"))
    merged = merged.merge(yolo[["frame", col]], on="frame")
    merged.rename(columns={col: f"{col}_yolo"}, inplace=True)

    # Extract aligned arrays
    gt_vals = merged[f"{col}_gt"].to_numpy()
    mp_vals = merged[f"{col}_mp"].to_numpy()
    yolo_vals = merged[f"{col}_yolo"].to_numpy()

    # === Calculate metrics ===
    metrics = {
        "Column": col,
        "MP_MAE": mean_absolute_error(gt_vals, mp_vals),
        "YOLO_MAE": mean_absolute_error(gt_vals, yolo_vals),
        "MP_RMSE": np.sqrt(mean_squared_error(gt_vals, mp_vals)),
        "YOLO_RMSE": np.sqrt(mean_squared_error(gt_vals, yolo_vals)),
        "MP_Corr": np.corrcoef(gt_vals, mp_vals)[0, 1],
        "YOLO_Corr": np.corrcoef(gt_vals, yolo_vals)[0, 1]
    }
    results.append(metrics)

    # === Plot ===
    plt.figure(figsize=(8, 5))
    plt.plot(merged["frame"], gt_vals, label="Ground Truth", marker='o', markersize=4)
    plt.plot(merged["frame"], mp_vals, label=f"Mediapipe\nMAE={metrics['MP_MAE']:.2f}, r={metrics['MP_Corr']:.2f}")
    plt.plot(merged["frame"], yolo_vals, label=f"YOLO\nMAE={metrics['YOLO_MAE']:.2f}, r={metrics['YOLO_Corr']:.2f}")

    # === Auto-detect type for labeling ===
    if "_angle" in col:
        ylabel = "Angle (°)"
        title_prefix = "Joint Angle"
    else:
        ylabel = "Distance (cm)"
        title_prefix = "Movement Distance"

    plt.title(f"{title_prefix}: {col.replace('_', ' ').title()}")
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Set Y-axis ticks based on data type
    if "_angle" in col:
        plt.yticks(np.arange(0, 190, 10))  # 0–180 degrees range typical
    else:
        max_val = max(gt_vals.max(), mp_vals.max(), yolo_vals.max())
        plt.yticks(np.arange(0, max_val + 10, 10))

    # Save the plot
    filename = os.path.join(OUT_DIR, f"{col}.png")
    plt.savefig(filename)
    print(f"✅ Saved plot: {filename}")
    plt.close()

# === Save metrics to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "model_errors.csv"), index=False)
print("\n📊 Metrics saved to:", os.path.join(OUT_DIR, "model_errors.csv"))

# === Display results summary ===
print("\n=== Summary of Errors ===")
print(results_df.round(3))

# === MPJPE calculation (Mean Per Joint Position Error) ===
try:
    # Joint CSVs
    GT_JOINTS = "joints_coordinates_test.csv"
    MP_JOINTS = "output/mediapipe_joints.csv"
    YOLO_JOINTS = "output/yolo_joints.csv"

    def load_joint_positions(path):
        dfj = pd.read_csv(path)
        pos = {}
        for _, r in dfj.iterrows():
            f = int(r['frame'])
            j = r['joint']
            pos.setdefault(f, {})[j] = (float(r['x']), float(r['y']))
        return pos

    gt_pos = load_joint_positions(GT_JOINTS)
    mp_pos = load_joint_positions(MP_JOINTS)
    yolo_pos = load_joint_positions(YOLO_JOINTS)

    # frames intersection
    common_frames_mp = sorted(set(gt_pos.keys()).intersection(mp_pos.keys()))
    common_frames_yolo = sorted(set(gt_pos.keys()).intersection(yolo_pos.keys()))

    def compute_mpjpe(gt_map, pred_map, frames):
        per_joint = {}
        all_errors = []
        for f in frames:
            gt_frame = gt_map.get(f, {})
            pred_frame = pred_map.get(f, {})
            joints_common = set(gt_frame.keys()).intersection(pred_frame.keys())
            for j in joints_common:
                gx, gy = gt_frame[j]
                px, py = pred_frame[j]
                err = np.sqrt((gx - px)**2 + (gy - py)**2)
                per_joint.setdefault(j, []).append(err)
                all_errors.append(err)
        # compute means
        per_joint_mean = {j: np.mean(vals) for j, vals in per_joint.items()} if per_joint else {}
        overall_mean = np.mean(all_errors) if all_errors else np.nan
        return overall_mean, per_joint_mean

    # default scale (cm per pixel) - adjust if you have a different reference
    BASKET_HEIGHT_CM = 7.5
    BASKET_PIXEL_HEIGHT = 35
    cm_per_pixel = BASKET_HEIGHT_CM / BASKET_PIXEL_HEIGHT

    mp_overall_px, mp_per_joint_px = compute_mpjpe(gt_pos, mp_pos, common_frames_mp)
    yolo_overall_px, yolo_per_joint_px = compute_mpjpe(gt_pos, yolo_pos, common_frames_yolo)

    # Save MPJPE summary
    rows = []
    rows.append({
        'method': 'MediaPipe', 'mpjpe_px': mp_overall_px, 'mpjpe_cm': (mp_overall_px * cm_per_pixel) if not np.isnan(mp_overall_px) else np.nan
    })
    rows.append({
        'method': 'YOLO', 'mpjpe_px': yolo_overall_px, 'mpjpe_cm': (yolo_overall_px * cm_per_pixel) if not np.isnan(yolo_overall_px) else np.nan
    })

    # per-joint rows
    for j, v in mp_per_joint_px.items():
        rows.append({'method': 'MediaPipe', 'joint': j, 'mpjpe_px': v, 'mpjpe_cm': v * cm_per_pixel})
    for j, v in yolo_per_joint_px.items():
        rows.append({'method': 'YOLO', 'joint': j, 'mpjpe_px': v, 'mpjpe_cm': v * cm_per_pixel})

    mpjpe_df = pd.DataFrame(rows)
    mpjpe_path = os.path.join(OUT_DIR, 'mpjpe_summary.csv')
    mpjpe_df.to_csv(mpjpe_path, index=False)
    print(f"\n✅ MPJPE summary saved: {mpjpe_path}")
    print(mpjpe_df)
except Exception as e:
    print("Could not compute MPJPE:", e)
