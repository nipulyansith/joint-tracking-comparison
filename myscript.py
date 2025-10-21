import cv2
import csv

video_path = "nipuledit2.mp4"
output_csv = "joints_coordinates_test.csv"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Total Frames:", total_frames)

# Joints in sequential order (added wrists)
JOINTS = [
    "right_elbow", "left_elbow",
    "right_knee", "left_knee",
    "right_shoulder", "left_shoulder",
    "right_hip", "left_hip",
    "right_wrist", "left_wrist"   # ✅ added wrists
]

coordinates = []
clicked_points = []
frame_idx = 0
basket_points = []  # store top & bottom of basket

# Resize factor for display
resize_factor = 1.2  # increased for larger display

# Mode control: first capture basket, then joints
mode = "basket"  # "basket" or "joints"

def mouse_callback(event, x, y, flags, param):
    global clicked_points, basket_points, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x / resize_factor)
        orig_y = int(y / resize_factor)

        if mode == "basket":
            basket_points.append((orig_x, orig_y))
            print(f"Clicked basket point {len(basket_points)}: {orig_x}, {orig_y}")
            if len(basket_points) == 2:
                basket_height = abs(basket_points[0][1] - basket_points[1][1])
                print(f"✅ Basket height (pixels): {basket_height}")
                mode = "joints"  # switch to joint labeling
        elif mode == "joints":
            if len(clicked_points) < len(JOINTS):
                clicked_points.append((orig_x, orig_y))
                joint_name = JOINTS[len(clicked_points) - 1]
                print(f"Clicked {joint_name}: {orig_x}, {orig_y}")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for display
    display = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

    # Draw basket points
    for i, (x, y) in enumerate(basket_points):
        cv2.circle(display, (int(x*resize_factor), int(y*resize_factor)), 5, (255, 255, 0), -1)
        cv2.putText(display, f"Basket-{i+1}", (int(x*resize_factor)+5, int(y*resize_factor)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw clicked joints
    for i, (x, y) in enumerate(clicked_points):
        cv2.circle(display, (int(x*resize_factor), int(y*resize_factor)), 5, (0, 0, 255), -1)
        cv2.putText(display, JOINTS[i], (int(x*resize_factor)+5, int(y*resize_factor)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(display, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if mode == "basket":
        cv2.putText(display, f"Click TOP & BOTTOM of basket", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(display, f"Clicked {len(clicked_points)}/{len(JOINTS)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display, "Press 'n' to save and go next, 'q' to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Frame", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n') and mode == "joints":
        # Save even if no joints clicked
        if len(clicked_points) > 0:
            for i, joint in enumerate(clicked_points):
                coordinates.append([frame_idx, JOINTS[i], joint[0], joint[1]])
            print(f"✅ Frame {frame_idx} saved with {len(clicked_points)} joints")
        else:
            print(f"➡️ Frame {frame_idx} skipped (no joints clicked)")
        clicked_points = []
        frame_idx += 1

    elif key == ord('q'):
        print("Exiting...")
        break

# Save CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "joint", "x", "y"])
    writer.writerows(coordinates)

cap.release()
cv2.destroyAllWindows()
print(f"✅ Saved {len(coordinates)} joint coordinates to {output_csv}")
