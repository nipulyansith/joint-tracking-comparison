# fixed_yolo_mediapipe_joints.py
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO

# --- SETTINGS ---
VIDEO_PATH = "nipuledit2.mp4"
JOINTS = [
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist"
]

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
MEDIAPIPE_CSV = os.path.join(OUT_DIR, "mediapipe_joints.csv")
YOLO_CSV = os.path.join(OUT_DIR, "yolo_joints.csv")
SNAPSHOT_DIR = os.path.join(OUT_DIR, "debug_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# --- Mediapipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- YOLO setup (pose model) ---
# Use a pose model file you have: e.g. "yolov8n-pose.pt" (downloaded or provided by ultralytics)
yolo_model = YOLO("yolov8n-pose.pt")

# --- COCO keypoint order (standard) ---
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
JOINT_TO_COCO_IDX = {
    "right_shoulder": 6,
    "right_elbow": 8,
    "right_wrist": 10,
    "left_shoulder": 5,
    "left_elbow": 7,
    "left_wrist": 9
}

# --- Prepare results containers ---
mediapipe_results = []
yolo_results = []

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
snap_frames = [0, max(0, frame_count - 1)]   # first and last frame for debugging
frame_idx = 0

print("Starting processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape

    # ----- Mediapipe -----
    mp_coords = {}  # store for this frame for debugging
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res = pose.process(image_rgb)
    if mp_res.pose_landmarks:
        lm = mp_res.pose_landmarks.landmark
        mp_map = {
            "right_elbow": lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
            "left_elbow": lm[mp_pose.PoseLandmark.LEFT_ELBOW],
            "right_knee": lm[mp_pose.PoseLandmark.RIGHT_KNEE],
            "left_knee": lm[mp_pose.PoseLandmark.LEFT_KNEE],
            "right_shoulder": lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            "left_shoulder": lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            "right_hip": lm[mp_pose.PoseLandmark.RIGHT_HIP],
            "left_hip": lm[mp_pose.PoseLandmark.LEFT_HIP],
            "right_wrist": lm[mp_pose.PoseLandmark.RIGHT_WRIST],
            "left_wrist": lm[mp_pose.PoseLandmark.LEFT_WRIST]
        }
        for joint_name in JOINTS:
            x = int(np.clip(mp_map[joint_name].x * w, 0, w - 1))
            y = int(np.clip(mp_map[joint_name].y * h, 0, h - 1))
            mediapipe_results.append({"frame": frame_idx, "joint": joint_name, "x": x, "y": y})
            mp_coords[joint_name] = (x, y)

    # ----- YOLO pose -----
    yolo_coords = {}
    try:
        yolo_out = yolo_model(frame, verbose=False)  # run model on full original frame
    except Exception as e:
        print("YOLO model error:", e)
        yolo_out = []

    # if a person is detected, keypoints should exist on the first result
    if len(yolo_out) > 0:
        res0 = yolo_out[0]
        if getattr(res0, "keypoints", None) is not None:
            # Try different ways to extract -> be defensive for torch vs numpy
            kp = None
            try:
                kp = res0.keypoints.xy[0].cpu().numpy()      # torch -> numpy (shape (17,2))
            except Exception:
                try:
                    kp = res0.keypoints.xy[0].numpy()         # numpy
                except Exception:
                    # fallback: convert keypoints object to numpy
                    try:
                        kp = np.array(res0.keypoints)[0][:, :2]
                    except Exception:
                        kp = None

            if kp is not None:
                # kp is (17,2) following COCO order
                for joint_name in JOINTS:
                    idx = JOINT_TO_COCO_IDX[joint_name]
                    x_f, y_f = kp[idx]
                    x_i = int(np.clip(round(float(x_f)), 0, w - 1))
                    y_i = int(np.clip(round(float(y_f)), 0, h - 1))
                    yolo_results.append({"frame": frame_idx, "joint": joint_name, "x": x_i, "y": y_i})
                    yolo_coords[joint_name] = (x_i, y_i)

    # ----- Debug snapshot (overlay both) -----
    if frame_idx in snap_frames:
        vis = frame.copy()
        # draw Mediapipe (blue) and YOLO (orange) keypoints and labels
        for j in JOINTS:
            if j in mp_coords:
                cv2.circle(vis, mp_coords[j], 5, (255, 0, 0), -1)   # blue
                cv2.putText(vis, f"mp:{j}", (mp_coords[j][0]+3, mp_coords[j][1]-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)
            if j in yolo_coords:
                cv2.circle(vis, yolo_coords[j], 5, (0,140,255), -1) # orange
                cv2.putText(vis, f"y:{j}", (yolo_coords[j][0]+3, yolo_coords[j][1]+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,140,255), 1, cv2.LINE_AA)
        snap_path = os.path.join(SNAPSHOT_DIR, f"frame_{frame_idx}_debug.png")
        cv2.imwrite(snap_path, vis)
        print(f"Saved debug snapshot: {snap_path}")

    frame_idx += 1

cap.release()
pose.close()

# --- Save CSVs ---
pd.DataFrame(mediapipe_results).to_csv(MEDIAPIPE_CSV, index=False)
pd.DataFrame(yolo_results).to_csv(YOLO_CSV, index=False)
print("Saved:", MEDIAPIPE_CSV, YOLO_CSV)
print("Done.")
