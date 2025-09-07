import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# === SETTINGS ===
VIDEO_FILE = "junior_side.mp4"  # path to your video
OUTPUT_FOLDER = Path("output")
OUTPUT_FOLDER.mkdir(exist_ok=True)
YOLO_CSV = OUTPUT_FOLDER / "yolo_joints.csv"

# List of joints we want
JOINTS = [
    "right_elbow", "left_elbow",
    "right_knee", "left_knee",
    "right_shoulder", "left_shoulder",
    "right_hip", "left_hip",
    "right_wrist", "left_wrist"
]

# === Load YOLOv8 pose model ===
model = YOLO("yolov8n-pose.pt")  # Make sure you have yolov8-pose model

# === Open video ===
cap = cv2.VideoCapture(VIDEO_FILE)
frame_idx = 0
results_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose detection
    results = model(frame, verbose=False)
    
    # YOLOv8 returns keypoints in results[0].keypoints.xy
    if results and len(results) > 0:
        for r in results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                kpts = r.keypoints.xy  # shape: (num_people, num_joints, 2)
                # Take first person detected
                person_kpts = kpts[0]  # shape: (num_joints, 2)

                # Map YOLO keypoints to our JOINTS list
                for i, joint in enumerate(JOINTS):
                    x, y = person_kpts[i]
                    results_list.append({
                        "frame": frame_idx,
                        "joint": joint,
                        "x": int(x),
                        "y": int(y)
                    })
            else:
                # No keypoints detected
                for joint in JOINTS:
                    results_list.append({
                        "frame": frame_idx,
                        "joint": joint,
                        "x": 0,
                        "y": 0
                    })
    else:
        # No person detected
        for joint in JOINTS:
            results_list.append({
                "frame": frame_idx,
                "joint": joint,
                "x": 0,
                "y": 0
            })

    frame_idx += 1

# Save to CSV
df = pd.DataFrame(results_list)
df.to_csv(YOLO_CSV, index=False)
print(f"✅ Saved YOLO joints to {YOLO_CSV}")
