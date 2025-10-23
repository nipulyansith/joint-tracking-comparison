import pandas as pd
import math

# === SETTINGS ===
CSV_FILE = "output/yolo_joints.csv"   # <-- change this to your YOLO csv
OUTPUT_FILE = "output/yolo_distances.csv"
BASKET_HEIGHT_CM = 17.6        # Real-world basket height in cm
BASKET_PIXEL_HEIGHT = 35       # Basket height in pixels (measured from clicks)

# === Load CSV ===
df = pd.read_csv(CSV_FILE)

# Calculate scale: cm per pixel
cm_per_pixel = BASKET_HEIGHT_CM / BASKET_PIXEL_HEIGHT
print(f"Scale: {cm_per_pixel:.4f} cm per pixel")

# Function to calculate distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * cm_per_pixel

# Define joint pairs we care about (static body proportions)
PAIRS = [
    ("left_shoulder", "right_shoulder"),   # Shoulder width
    ("left_hip", "right_hip"),             # Hip width
    ("left_shoulder", "left_elbow"),       # Left arm
    ("right_shoulder", "right_elbow"),     # Right arm
    ("left_shoulder", "left_hip"),         # Left body side
    ("right_shoulder", "right_hip")        # Right body side
]

# === Collect results per frame ===
results = []
frames_sorted = sorted(df['frame'].unique())  # ensure chronological order
prev_positions = {}  # store previous positions for movement

for frame in frames_sorted:
    frame_data = df[df['frame'] == frame]
    row = {"frame": frame}

    # --- Static distances (body proportions) ---
    for j1, j2 in PAIRS:
        p1 = frame_data[frame_data['joint'] == j1][['x','y']].values
        p2 = frame_data[frame_data['joint'] == j2][['x','y']].values

        if len(p1) > 0 and len(p2) > 0:
            dist_cm = distance(p1[0], p2[0])
            row[f"{j1}_to_{j2}_cm"] = dist_cm

            # ✅ Assign shoulder width directly if this pair is shoulders
            if j1 == "left_shoulder" and j2 == "right_shoulder":
                row["shoulder_width"] = dist_cm

        else:
            row[f"{j1}_to_{j2}_cm"] = None
            if j1 == "left_shoulder" and j2 == "right_shoulder":
                row["shoulder_width"] = None

    # --- Movement tracking (wrist, elbow, shoulder, knee) ---
    moving_joints = [
        "left_wrist", "right_wrist",
        "left_elbow", "right_elbow",
        "left_shoulder", "right_shoulder",
        "left_knee", "right_knee"
    ]

    for joint in moving_joints:
        p = frame_data[frame_data['joint'] == joint][['x','y']].values
        if len(p) > 0:
            p = p[0]
            if joint in prev_positions:
                row[f"{joint}_movement_cm"] = distance(p, prev_positions[joint])
            else:
                row[f"{joint}_movement_cm"] = 0.0  # no movement for first frame
            prev_positions[joint] = p
        else:
            row[f"{joint}_movement_cm"] = None

    results.append(row)

# === Save results ===
results_df = pd.DataFrame(results)

canonical_cols = [
    'frame',
    'left_shoulder_to_right_shoulder_cm',
    'left_hip_to_right_hip_cm',
    'left_shoulder_to_left_elbow_cm',
    'right_shoulder_to_right_elbow_cm',
    'left_shoulder_to_left_hip_cm',
    'right_shoulder_to_right_hip_cm',
    # Movements
    'left_wrist_movement_cm',
    'right_wrist_movement_cm',
    'left_elbow_movement_cm',
    'right_elbow_movement_cm',
    'left_shoulder_movement_cm',
    'right_shoulder_movement_cm',
    'left_knee_movement_cm',
    'right_knee_movement_cm',
    # Derived / summary
    'shoulder_width'
]

# Ensure all expected columns exist even if missing in results
for col in canonical_cols:
    if col not in results_df.columns:
        results_df[col] = None

results_df = results_df[canonical_cols]
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Distances + movements (shoulders, elbows, wrists, knees) saved to {OUTPUT_FILE}")
print(df['frame'].unique())
print(len(df['frame'].unique()))
