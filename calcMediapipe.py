import pandas as pd
import math

# === SETTINGS ===
CSV_FILE = "output/mediapipe_joints.csv"   # <-- Mediapipe joints CSV
OUTPUT_FILE = "output/mediapipe_distances.csv"
BASKET_HEIGHT_CM = 17.6        # Real-world basket height in cm
BASKET_PIXEL_HEIGHT = 35    # Basket height in pixels (measured from clicks)

# === Load CSV ===
df = pd.read_csv(CSV_FILE)

# Calculate scale: cm per pixel
cm_per_pixel = BASKET_HEIGHT_CM / BASKET_PIXEL_HEIGHT
print(f"Scale: {cm_per_pixel:.4f} cm per pixel")

# Function to calculate distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * cm_per_pixel

# Ground truth measurements (in cm)
GROUND_TRUTH = {
    "shoulder_to_elbow": 28.0,  # cm
    "elbow_to_wrist": 27.0      # cm
}

# Define joint pairs we care about (static body proportions)
PAIRS = [
    ("left_shoulder", "left_elbow"),       # Left upper arm
    ("right_shoulder", "right_elbow"),     # Right upper arm
    ("left_elbow", "left_wrist"),         # Left forearm
    ("right_elbow", "right_wrist"),       # Right forearm
    ("left_shoulder", "right_shoulder"),   # Shoulder width
    ("left_hip", "right_hip"),            # Hip width
    ("left_shoulder", "left_hip"),        # Left body side
    ("right_shoulder", "right_hip")       # Right body side
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
            row[f"{j1}_to_{j2}_cm"] = distance(p1[0], p2[0])
        else:
            row[f"{j1}_to_{j2}_cm"] = None

    # --- Movement tracking (wrist + elbow motion) ---
    for joint in ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]:
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

    # also expose shoulder width as a convenience column (left_shoulder_to_right_shoulder_cm)
    row['shoulder_width'] = row.get('left_shoulder_to_right_shoulder_cm', None)

    results.append(row)

# Save results
results_df = pd.DataFrame(results)

# Ensure consistent column order across calculators
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
    'shoulder_width'
]

# Reindex will add any missing columns with NaN and order them consistently
results_df = results_df.reindex(columns=canonical_cols)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Distances + movements saved to {OUTPUT_FILE}")
print(f"Frames processed: {len(frames_sorted)}")
