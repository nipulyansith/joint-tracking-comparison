import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ground truth values
GROUND_TRUTH = {
    "shoulder_to_elbow": 28.0,  # cm
    "elbow_to_wrist": 27.0      # cm
}

# Load MediaPipe results
mp_df = pd.read_csv("output/mediapipe_distances.csv")

# Load YOLO results
yolo_df = pd.read_csv("output/yolo_distances.csv")

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Arm Length Measurements: MediaPipe vs YOLO vs Ground Truth', fontsize=16)

# Colors
colors = {
    'MediaPipe': 'blue',
    'YOLO': 'green',
    'Ground Truth': 'red'
}

# Plot Left Arm Upper (shoulder to elbow)
def plot_measurements(ax, mp_data, yolo_data, joint_name, ground_truth):
    ax.plot(mp_data, label='MediaPipe', color=colors['MediaPipe'], alpha=0.6)
    ax.plot(yolo_data, label='YOLO', color=colors['YOLO'], alpha=0.6)
    ax.axhline(y=ground_truth, color=colors['Ground Truth'], linestyle='--', 
               label=f'Ground Truth ({ground_truth}cm)')
    ax.set_ylabel('Length (cm)')
    ax.set_xlabel('Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Left Arm
plot_measurements(
    axes[0,0],
    mp_df['left_shoulder_to_left_elbow_cm'],
    yolo_df['left_shoulder_to_left_elbow_cm'],
    'Left Upper Arm',
    GROUND_TRUTH['shoulder_to_elbow']
)
axes[0,0].set_title('Left Upper Arm (Shoulder to Elbow)')

plot_measurements(
    axes[0,1],
    mp_df['left_elbow_to_left_wrist_cm'],
    yolo_df['left_elbow_to_left_wrist_cm'],
    'Left Forearm',
    GROUND_TRUTH['elbow_to_wrist']
)
axes[0,1].set_title('Left Forearm (Elbow to Wrist)')

# Right Arm
plot_measurements(
    axes[1,0],
    mp_df['right_shoulder_to_right_elbow_cm'],
    yolo_df['right_shoulder_to_right_elbow_cm'],
    'Right Upper Arm',
    GROUND_TRUTH['shoulder_to_elbow']
)
axes[1,0].set_title('Right Upper Arm (Shoulder to Elbow)')

plot_measurements(
    axes[1,1],
    mp_df['right_elbow_to_right_wrist_cm'],
    yolo_df['right_elbow_to_right_wrist_cm'],
    'Right Forearm',
    GROUND_TRUTH['elbow_to_wrist']
)
axes[1,1].set_title('Right Forearm (Elbow to Wrist)')

# Adjust layout and save
plt.tight_layout()
plt.savefig('output/arm_length_comparison.png')
plt.show()

# Calculate average errors
methods = ['MediaPipe', 'YOLO']
segments = [
    ('left_shoulder_to_left_elbow_cm', 'shoulder_to_elbow'),
    ('left_elbow_to_left_wrist_cm', 'elbow_to_wrist'),
    ('right_shoulder_to_right_elbow_cm', 'shoulder_to_elbow'),
    ('right_elbow_to_right_wrist_cm', 'elbow_to_wrist')
]

print("\nAverage Errors (cm):")
print("-" * 50)
for method, df in [('MediaPipe', mp_df), ('YOLO', yolo_df)]:
    print(f"\n{method}:")
    for col, truth_key in segments:
        mean_val = df[col].mean()
        error = abs(mean_val - GROUND_TRUTH[truth_key])
        print(f"{col:<35}: {error:.2f} cm")
