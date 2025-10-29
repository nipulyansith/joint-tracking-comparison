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

# === Columns to plot and evaluate ===
columns_to_plot = [
    "left_wrist_movement_cm",
    "right_wrist_movement_cm",
    "left_elbow_movement_cm",
    "right_elbow_movement_cm",
    "left_shoulder_movement_cm",
    "right_shoulder_movement_cm",
    "left_knee_movement_cm",
    "right_knee_movement_cm"
]

# === Prepare results list ===
results = []

# === Loop through each selected column ===
for col in columns_to_plot:
    if col not in gt.columns:
        print(f"⚠️ Column '{col}' not found in Ground Truth file. Skipping...")
        continue

    # Ensure same frame alignment
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
    plt.plot(merged["frame"], gt_vals, label="Ground Truth", marker='o')
    plt.plot(merged["frame"], mp_vals, label=f"Mediapipe\nMAE={metrics['MP_MAE']:.2f}, r={metrics['MP_Corr']:.2f}")
    plt.plot(merged["frame"], yolo_vals, label=f"YOLO\nMAE={metrics['YOLO_MAE']:.2f}, r={metrics['YOLO_Corr']:.2f}")

    plt.title(col.replace("_cm", "").replace("_", " ").title())
    plt.xlabel("Frame")
    plt.ylabel("Distance (cm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Set Y-axis ticks in multiples of 10
    max_val = max(gt_vals.max(), mp_vals.max(), yolo_vals.max())
    y_ticks = np.arange(0, max_val + 10, 10)
    plt.yticks(y_ticks)

    # Save the plot
    filename = os.path.join(OUT_DIR, f"{col}.png")
    plt.savefig(filename)
    print(f"✅ Saved plot: {filename}")
    plt.close()

# === Save metrics to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "model_errors.csv"), index=False)
print("\n📊 Metrics saved to:", os.path.join(OUT_DIR, "model_errors.csv"))

# Display results summary
print("\n=== Summary of Errors ===")
print(results_df.round(3))
