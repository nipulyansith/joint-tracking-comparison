import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_angle(shoulder, elbow, wrist):
    """Calculate angle between three points (shoulder-elbow-wrist)"""
    vector1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
    vector2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
    
    # Calculate the angle using dot product
    dot_product = np.dot(vector1, vector2)
    norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    
    # Avoid division by zero
    if norms == 0:
        return 0
    
    cos_angle = dot_product / norms
    # Ensure the value is in valid range for arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# File paths
GT_JOINTS_FILE = "joints_coordinates_test.csv"  # Ground truth joints
MP_JOINTS_FILE = "output/mediapipe_joints.csv"
YOLO_JOINTS_FILE = "output/yolo_joints.csv"

def calculate_gt_angles(joints_df):
    """Calculate angles from ground truth joint coordinates"""
    angles = {"frame": [], "left_elbow_angle_deg": [], "right_elbow_angle_deg": []}
    
    for frame in joints_df['frame'].unique():
        frame_data = joints_df[joints_df['frame'] == frame]
        
        # Get coordinates for left arm
        left_shoulder = frame_data[frame_data['joint'] == 'left_shoulder'][['x', 'y']].values[0]
        left_elbow = frame_data[frame_data['joint'] == 'left_elbow'][['x', 'y']].values[0]
        left_wrist = frame_data[frame_data['joint'] == 'left_wrist'][['x', 'y']].values[0]
        
        # Get coordinates for right arm
        right_shoulder = frame_data[frame_data['joint'] == 'right_shoulder'][['x', 'y']].values[0]
        right_elbow = frame_data[frame_data['joint'] == 'right_elbow'][['x', 'y']].values[0]
        right_wrist = frame_data[frame_data['joint'] == 'right_wrist'][['x', 'y']].values[0]
        
        # Calculate angles
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        angles["frame"].append(frame)
        angles["left_elbow_angle_deg"].append(left_angle)
        angles["right_elbow_angle_deg"].append(right_angle)
    
    return pd.DataFrame(angles)
OUT_DIR = "plottedImages"

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# Load joint coordinates and calculate angles
gt_joints = pd.read_csv(GT_JOINTS_FILE)
mp_joints = pd.read_csv(MP_JOINTS_FILE)
yolo_joints = pd.read_csv(YOLO_JOINTS_FILE)

# Calculate ground truth angles
gt = calculate_gt_angles(gt_joints)

# Calculate angles for MediaPipe
mp_angles = {"frame": [], "left_elbow_angle_deg": [], "right_elbow_angle_deg": []}

for frame in mp_joints['frame'].unique():
    frame_data = mp_joints[mp_joints['frame'] == frame]
    
    # Get coordinates for left arm
    left_shoulder = frame_data[frame_data['joint'] == 'left_shoulder'][['x', 'y']].values[0]
    left_elbow = frame_data[frame_data['joint'] == 'left_elbow'][['x', 'y']].values[0]
    left_wrist = frame_data[frame_data['joint'] == 'left_wrist'][['x', 'y']].values[0]
    
    # Get coordinates for right arm
    right_shoulder = frame_data[frame_data['joint'] == 'right_shoulder'][['x', 'y']].values[0]
    right_elbow = frame_data[frame_data['joint'] == 'right_elbow'][['x', 'y']].values[0]
    right_wrist = frame_data[frame_data['joint'] == 'right_wrist'][['x', 'y']].values[0]
    
    # Calculate angles
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    mp_angles["frame"].append(frame)
    mp_angles["left_elbow_angle_deg"].append(left_angle)
    mp_angles["right_elbow_angle_deg"].append(right_angle)

# Convert to DataFrame
mp_angles_df = pd.DataFrame(mp_angles)

# Calculate angles for YOLO
yolo_angles = {"frame": [], "left_elbow_angle_deg": [], "right_elbow_angle_deg": []}

for frame in yolo_joints['frame'].unique():
    frame_data = yolo_joints[yolo_joints['frame'] == frame]
    
    # Get coordinates for left arm
    left_shoulder = frame_data[frame_data['joint'] == 'left_shoulder'][['x', 'y']].values[0]
    left_elbow = frame_data[frame_data['joint'] == 'left_elbow'][['x', 'y']].values[0]
    left_wrist = frame_data[frame_data['joint'] == 'left_wrist'][['x', 'y']].values[0]
    
    # Get coordinates for right arm
    right_shoulder = frame_data[frame_data['joint'] == 'right_shoulder'][['x', 'y']].values[0]
    right_elbow = frame_data[frame_data['joint'] == 'right_elbow'][['x', 'y']].values[0]
    right_wrist = frame_data[frame_data['joint'] == 'right_wrist'][['x', 'y']].values[0]
    
    # Calculate angles
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    yolo_angles["frame"].append(frame)
    yolo_angles["left_elbow_angle_deg"].append(left_angle)
    yolo_angles["right_elbow_angle_deg"].append(right_angle)

# Convert to DataFrame
yolo_angles_df = pd.DataFrame(yolo_angles)

# Plot angles
angles_to_plot = ['left_elbow_angle_deg', 'right_elbow_angle_deg']

for angle in angles_to_plot:
    plt.figure(figsize=(10, 6))
    
    # Plot ground truth
    plt.plot(gt["frame"], gt[angle], label="Ground Truth", marker='o', color='green', linestyle='-', linewidth=2)
    
    # Plot MediaPipe
    plt.plot(mp_angles_df["frame"], mp_angles_df[angle], 
             label="MediaPipe", marker='s', color='blue', linestyle='--')
    
    # Plot YOLO
    plt.plot(yolo_angles_df["frame"], yolo_angles_df[angle], 
             label="YOLO", marker='^', color='red', linestyle=':')
    
    plt.figure(figsize=(12, 6))
    
    # Plot with enhanced visibility
    plt.plot(gt["frame"], gt[angle], 'go-', label="Ground Truth", 
             linewidth=2, markersize=10, markerfacecolor='white')
    
    plt.plot(mp_angles_df["frame"], mp_angles_df[angle], 'bs--', 
             label="MediaPipe", linewidth=1.5, markersize=8, markerfacecolor='white')
    
    plt.plot(yolo_angles_df["frame"], yolo_angles_df[angle], 'r^:', 
             label="YOLO", linewidth=1.5, markersize=8, markerfacecolor='white')
    
    plt.title(f"{angle.replace('_', ' ').title()}\nComparison of Different Methods", 
              pad=20, fontsize=12)
    plt.xlabel("Frame Number", fontsize=10)
    plt.ylabel("Angle (degrees)", fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize legend
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set consistent y-axis range for angles
    plt.ylim(0, 180)
    
    # Set x-axis to show all frame numbers
    all_frames = sorted(list(set(list(gt['frame']) + 
                               list(mp_angles_df['frame']) + 
                               list(yolo_angles_df['frame']))))
    plt.xticks(all_frames, rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    filename = os.path.join(OUT_DIR, f"{angle}_comparison.png")
    plt.savefig(filename)
    print(f"✅ Saved plot: {filename}")
    plt.close()

# Calculate metrics for each method
for angle in angles_to_plot:
    print(f"\n{'='*50}")
    print(f"Metrics for {angle.replace('_', ' ').title()}")
    print(f"{'='*50}")
    
    # Get ground truth frames
    gt_frames = gt['frame'].unique()
    
    # Filter MediaPipe and YOLO data to match ground truth frames
    mp_filtered = mp_angles_df[mp_angles_df['frame'].isin(gt_frames)]
    yolo_filtered = yolo_angles_df[yolo_angles_df['frame'].isin(gt_frames)]
    
    # Calculate metrics for MediaPipe
    mp_mae = np.mean(np.abs(gt[angle].values - mp_filtered[angle].values))
    mp_rmse = np.sqrt(np.mean((gt[angle].values - mp_filtered[angle].values)**2))
    mp_corr = np.corrcoef(gt[angle].values, mp_filtered[angle].values)[0,1]
    
    # Calculate metrics for YOLO
    yolo_mae = np.mean(np.abs(gt[angle].values - yolo_filtered[angle].values))
    yolo_rmse = np.sqrt(np.mean((gt[angle].values - yolo_filtered[angle].values)**2))
    yolo_corr = np.corrcoef(gt[angle].values, yolo_filtered[angle].values)[0,1]
    
    print("\nMediaPipe Metrics:")
    print(f"  Mean Absolute Error (MAE): {mp_mae:.2f}°")
    print(f"  Root Mean Square Error (RMSE): {mp_rmse:.2f}°")
    print(f"  Correlation coefficient (r): {mp_corr:.3f}")
    
    print("\nYOLO Metrics:")
    print(f"  Mean Absolute Error (MAE): {yolo_mae:.2f}°")
    print(f"  Root Mean Square Error (RMSE): {yolo_rmse:.2f}°")
    print(f"  Correlation coefficient (r): {yolo_corr:.3f}")
    
    # Add frame coverage information
    print(f"\nFrame Coverage:")
    print(f"  Ground Truth: frames {min(gt_frames)} to {max(gt_frames)} ({len(gt_frames)} frames)")
    print(f"  MediaPipe: frames {min(mp_angles_df['frame'])} to {max(mp_angles_df['frame'])} ({len(mp_angles_df)} frames)")
    print(f"  YOLO: frames {min(yolo_angles_df['frame'])} to {max(yolo_angles_df['frame'])} ({len(yolo_angles_df)} frames)")
