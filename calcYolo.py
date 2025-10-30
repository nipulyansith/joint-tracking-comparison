import pandas as pd
import math

# === SETTINGS (change these 2 lines only) ===
CSV_FILE = "output/yolo_joints.csv"       # Input joint coordinates file
OUTPUT_FILE = "output/yolo_distances.csv" # Output distance file

# === CONSTANTS ===
BASKET_HEIGHT_CM = 33.5      # Real-world basket height in cm
BASKET_PIXEL_HEIGHT = 192    # Basket height in pixels (from manual measurement)

# === Load CSV ===
df = pd.read_csv(CSV_FILE)

# Calculate scale: cm per pixel
cm_per_pixel = BASKET_HEIGHT_CM / BASKET_PIXEL_HEIGHT
print(f"Scale: {cm_per_pixel:.4f} cm per pixel")

# === Function to calculate distance between two points ===
def distance(p1, p2):
    """Euclidean distance between two (x,y) points in cm"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * cm_per_pixel

# === Function to calculate angle (in degrees) at point B given three points A-B-C ===
def angle_at_point(A, B, C):
    """Calculate the angle at point B (A-B-C) in degrees"""
    try:
        BA = (A[0] - B[0], A[1] - B[1])
        BC = (C[0] - B[0], C[1] - B[1])
        dot_product = BA[0]*BC[0] + BA[1]*BC[1]
        mag_BA = math.sqrt(BA[0]**2 + BA[1]**2)
        mag_BC = math.sqrt(BC[0]**2 + BC[1]**2)
        if mag_BA == 0 or mag_BC == 0:
            return None
        cos_angle = max(-1, min(1, dot_product / (mag_BA * mag_BC)))  # Clamp to avoid errors
        angle_deg = math.degrees(math.acos(cos_angle))
        return angle_deg
    except:
        return None

# === Define joint pairs (fixed body segments) ===
PAIRS = [
    ("left_shoulder", "right_shoulder"),  # Shoulder width
    ("left_hip", "right_hip"),            # Hip width
    ("left_shoulder", "left_elbow"),      # Left upper arm
    ("right_shoulder", "right_elbow"),    # Right upper arm
    ("left_elbow", "left_wrist"),         # Left forearm
    ("right_elbow", "right_wrist"),       # Right forearm
    ("left_shoulder", "left_hip"),        # Left body side
    ("right_shoulder", "right_hip")       # Right body side
]

# === Movement joints (to track motion per frame) ===
MOVING_JOINTS = [
    "left_wrist", "right_wrist",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_knee", "right_knee"
]

# === Collect results ===
results = []
frames_sorted = sorted(df['frame'].unique())
prev_positions = {}  # store last known positions

for frame in frames_sorted:
    frame_data = df[df['frame'] == frame]
    row = {"frame": frame}

    # --- Static distances ---
    for j1, j2 in PAIRS:
        p1 = frame_data[frame_data['joint'] == j1][['x', 'y']].values
        p2 = frame_data[frame_data['joint'] == j2][['x', 'y']].values

        if len(p1) > 0 and len(p2) > 0:
            dist_cm = distance(p1[0], p2[0])
            row[f"{j1}_to_{j2}_cm"] = dist_cm
            if j1 == "left_shoulder" and j2 == "right_shoulder":
                row["shoulder_width"] = dist_cm
        else:
            row[f"{j1}_to_{j2}_cm"] = None
            if j1 == "left_shoulder" and j2 == "right_shoulder":
                row["shoulder_width"] = None

    # --- Movement tracking ---
    for joint in MOVING_JOINTS:
        p = frame_data[frame_data['joint'] == joint][['x', 'y']].values
        if len(p) > 0:
            p = p[0]
            if joint in prev_positions:
                row[f"{joint}_movement_cm"] = distance(p, prev_positions[joint])
            else:
                row[f"{joint}_movement_cm"] = 0.0  # first frame = no movement
            prev_positions[joint] = p
        else:
            row[f"{joint}_movement_cm"] = None

    # --- Elbow Angles (wrist → elbow → shoulder) ---
    left_points = {
        "wrist": frame_data[frame_data['joint'] == "left_wrist"][['x', 'y']].values,
        "elbow": frame_data[frame_data['joint'] == "left_elbow"][['x', 'y']].values,
        "shoulder": frame_data[frame_data['joint'] == "left_shoulder"][['x', 'y']].values
    }
    right_points = {
        "wrist": frame_data[frame_data['joint'] == "right_wrist"][['x', 'y']].values,
        "elbow": frame_data[frame_data['joint'] == "right_elbow"][['x', 'y']].values,
        "shoulder": frame_data[frame_data['joint'] == "right_shoulder"][['x', 'y']].values
    }

    # Left elbow angle
    if all(len(left_points[j]) > 0 for j in left_points):
        row["left_elbow_angle_deg"] = angle_at_point(
            left_points["wrist"][0], left_points["elbow"][0], left_points["shoulder"][0]
        )
    else:
        row["left_elbow_angle_deg"] = None

    # Right elbow angle
    if all(len(right_points[j]) > 0 for j in right_points):
        row["right_elbow_angle_deg"] = angle_at_point(
            right_points["wrist"][0], right_points["elbow"][0], right_points["shoulder"][0]
        )
    else:
        row["right_elbow_angle_deg"] = None

    results.append(row)

# === Convert to DataFrame ===
results_df = pd.DataFrame(results)

# === Ensure consistent column order ===
canonical_cols = [
    'frame',
    'left_shoulder_to_right_shoulder_cm',
    'left_hip_to_right_hip_cm',
    'left_shoulder_to_left_elbow_cm',
    'right_shoulder_to_right_elbow_cm',
    'left_elbow_to_left_wrist_cm',
    'right_elbow_to_right_wrist_cm',
    'left_shoulder_to_left_hip_cm',
    'right_shoulder_to_right_hip_cm',
    'left_wrist_movement_cm',
    'right_wrist_movement_cm',
    'left_elbow_movement_cm',
    'right_elbow_movement_cm',
    'left_shoulder_movement_cm',
    'right_shoulder_movement_cm',
    'left_knee_movement_cm',
    'right_knee_movement_cm',
    'shoulder_width',
    'left_elbow_angle_deg',
    'right_elbow_angle_deg'
]

for col in canonical_cols:
    if col not in results_df.columns:
        results_df[col] = None

results_df = results_df[canonical_cols]

# === Save to CSV ===
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Distances + movements + elbow angles saved to {OUTPUT_FILE}")
print(f"Frames processed: {len(frames_sorted)}")
