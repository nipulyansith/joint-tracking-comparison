# fixed_yolo_mediapipe_joints_full.py
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO

# --- SETTINGS ---
VIDEO_PATH = "nipuledit2.mp4"
FRAME_STEP = 10  # process every 10th frame (change easily)

JOINTS = [
    "right_shoulder", "left_shoulder",
    "right_elbow", "left_elbow",
    "right_wrist", "left_wrist",
    "right_hip", "left_hip",
    "right_knee", "left_knee"
]

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
MEDIAPIPE_CSV = os.path.join(OUT_DIR, "mediapipe_joints.csv")
YOLO_CSV = os.path.join(OUT_DIR, "yolo_joints.csv")
SNAPSHOT_DIR = os.path.join(OUT_DIR, "debug_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# --- Mediapipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.1, min_tracking_confidence=0.1)

# --- YOLO setup (pose model) ---
yolo_model = YOLO("yolov8n-pose.pt")

# --- COCO keypoint order (YOLOv8 pose model standard) ---
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
JOINT_TO_COCO_IDX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
}

# --- Results containers ---
mediapipe_results = []
yolo_results = []

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
snap_frames = [0, max(0, frame_count - 1)]  # for debugging
frame_idx = 0
processed_idx = 0

print(f"Starting processing... total frames: {frame_count}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # process every Nth frame only
    if frame_idx % FRAME_STEP != 0:
        frame_idx += 1
        continue

    h, w, _ = frame.shape

    # ----- Mediapipe -----
    mp_coords = {}
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res = pose.process(image_rgb)
    if mp_res.pose_landmarks:
        lm = mp_res.pose_landmarks.landmark
        for joint_name in JOINTS:
            try:
                lm_point = getattr(mp_pose.PoseLandmark, joint_name.upper())
                x = int(np.clip(lm[lm_point].x * w, 0, w - 1))
                y = int(np.clip(lm[lm_point].y * h, 0, h - 1))
                mediapipe_results.append({"frame": frame_idx, "joint": joint_name, "x": x, "y": y})
                mp_coords[joint_name] = (x, y)
            except Exception:
                continue  # skip if joint missing

    # ----- YOLO Pose -----
    yolo_coords = {}
    try:
        yolo_out = yolo_model(frame, verbose=False)
        if len(yolo_out) > 0 and getattr(yolo_out[0], "keypoints", None) is not None:
            kp = yolo_out[0].keypoints.xy[0].cpu().numpy()  # shape (17,2)
            for joint_name in JOINTS:
                if joint_name in JOINT_TO_COCO_IDX:
                    idx = JOINT_TO_COCO_IDX[joint_name]
                    x_f, y_f = kp[idx]
                    x_i = int(np.clip(round(float(x_f)), 0, w - 1))
                    y_i = int(np.clip(round(float(y_f)), 0, h - 1))
                    yolo_results.append({"frame": frame_idx, "joint": joint_name, "x": x_i, "y": y_i})
                    yolo_coords[joint_name] = (x_i, y_i)
    except Exception as e:
        print("YOLO model error:", e)

    # ----- Debug snapshot -----
    if frame_idx in snap_frames:
        vis = frame.copy()
        for j in JOINTS:
            if j in mp_coords:
                cv2.circle(vis, mp_coords[j], 5, (255, 0, 0), -1)
                cv2.putText(vis, f"mp:{j}", (mp_coords[j][0]+3, mp_coords[j][1]-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
            if j in yolo_coords:
                cv2.circle(vis, yolo_coords[j], 5, (0,140,255), -1)
                cv2.putText(vis, f"y:{j}", (yolo_coords[j][0]+3, yolo_coords[j][1]+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,140,255), 1)
        snap_path = os.path.join(SNAPSHOT_DIR, f"frame_{frame_idx}_debug.png")
        cv2.imwrite(snap_path, vis)
        print(f"Saved debug snapshot: {snap_path}")

    processed_idx += 1
    frame_idx += 1

cap.release()
pose.close()

# --- Save CSVs ---
pd.DataFrame(mediapipe_results).to_csv(MEDIAPIPE_CSV, index=False)
pd.DataFrame(yolo_results).to_csv(YOLO_CSV, index=False)
print(f"\nSaved:\n- {MEDIAPIPE_CSV}\n- {YOLO_CSV}")
print("Done.")
