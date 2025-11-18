import mediapipe as mp
import cv2
import numpy as np
import time
import pickle
import warnings

warnings.filterwarnings('ignore')

# Load the trained ML model from the pickle file.
with open('yoga_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe drawing and pose modules.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Precompute CLAHE and gamma correction table.
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
GAMMA = 1.2
invGamma = 1.0 / GAMMA
gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")

def preprocess_frame(frame):
    """
    Enhance the frame using CLAHE and gamma correction.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = CLAHE.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    final = cv2.LUT(enhanced, gamma_table)
    return final

def compute_joint_angle_from_coords(landmarks, i, j, k):
    """
    Compute the angle (in degrees) at landmark j using landmarks i, j, k.
    landmarks: numpy array shape (N, 2) where each row is [x, y].
    i, j, k: indices of the landmarks.
    """
    a = landmarks[i]
    b = landmarks[j]
    c = landmarks[k]
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def draw_skeleton(frame, landmarks, connections, color, thickness, alpha=0.6):
    """
    Draw a skeleton on the frame using only lines (no circles) and exclude face connections.
    """
    overlay = frame.copy()
    body_connections = [
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),           # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]
    for connection in body_connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
        end_point = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
        cv2.line(overlay, start_point, end_point, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def extract_landmarks(results):
    """
    Extract pose landmarks and return them as a flattened array.
    """
    row = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        row.extend([0] * (33 * 4))
    return row

# Processing resolution for efficiency.
PROCESS_WIDTH, PROCESS_HEIGHT = 640, 480

cap = cv2.VideoCapture(0)
prev_time = 0

# Global variable for smoothing landmarks.
smoothed_landmarks = None
smoothing_factor = 0.7  # Adjust this factor (0.0 no smoothing, closer to 1.0 smoother but laggier)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        display_frame = frame_resized.copy()

        proc_frame = preprocess_frame(frame_resized)
        proc_frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        proc_frame_rgb.flags.writeable = False

        results = pose.process(proc_frame_rgb)

        # Default values.
        skeleton_color = (0, 0, 255)  # Red for incorrect.
        pose_text = "No Pose Detected"
        accuracy_text = ""

        if results.pose_landmarks:
            new_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
            if smoothed_landmarks is None:
                smoothed_landmarks = new_landmarks
            else:
                smoothed_landmarks = smoothing_factor * smoothed_landmarks + (1 - smoothing_factor) * new_landmarks

            for i, lm in enumerate(results.pose_landmarks.landmark):
                lm.x = smoothed_landmarks[i, 0]
                lm.y = smoothed_landmarks[i, 1]
                lm.z = smoothed_landmarks[i, 2]
                lm.visibility = smoothed_landmarks[i, 3]

            landmarks = results.pose_landmarks.landmark
            coords = np.array([[lm.x * PROCESS_WIDTH, lm.y * PROCESS_HEIGHT] for lm in landmarks])

            # --- Warrior Pose Logic ---
            warrior_left_elbow_angle = compute_joint_angle_from_coords(coords, 11, 13, 15)
            warrior_right_elbow_angle = compute_joint_angle_from_coords(coords, 12, 14, 16)
            warrior_left_knee_angle = compute_joint_angle_from_coords(coords, 23, 25, 27)
            warrior_right_knee_angle = compute_joint_angle_from_coords(coords, 24, 26, 28)
            warrior_left_arm_angle = compute_joint_angle_from_coords(coords, 23, 11, 13)
            warrior_right_arm_angle = compute_joint_angle_from_coords(coords, 24, 12, 14)

            warrior_elbows_ok = (150 <= warrior_left_elbow_angle <= 190) and (150 <= warrior_right_elbow_angle <= 190)
            warrior_knees_ok = ((110 <= warrior_left_knee_angle <= 150) and (150 <= warrior_right_knee_angle <= 190)) or \
                            ((110 <= warrior_right_knee_angle <= 150) and (150 <= warrior_left_knee_angle <= 190))
            warrior_arms_alignment_ok = (80 <= warrior_left_arm_angle <= 100) and (80 <= warrior_right_arm_angle <= 100)
            warrior_pose = warrior_elbows_ok and warrior_knees_ok and warrior_arms_alignment_ok


            # Raised Hands Pose Logic
            raised_hands_left_shoulder_angle = compute_joint_angle_from_coords(coords, 23, 11, 13)
            raised_hands_right_shoulder_angle = compute_joint_angle_from_coords(coords, 24, 12, 14)
            raised_hands_left_hip_angle = compute_joint_angle_from_coords(coords, 25, 23, 11)
            raised_hands_right_hip_angle = compute_joint_angle_from_coords(coords, 26, 24, 12)

            raised_hands_raised_shoulders_ok = (160 <= raised_hands_left_shoulder_angle <= 180) and (160 <= raised_hands_right_shoulder_angle <= 180)
            raised_hands_raised_hips_ok = (raised_hands_left_hip_angle < 180) and (raised_hands_right_hip_angle < 180)
            raised_hands_hands_ok = (coords[15][1] < coords[11][1]) and (coords[16][1] < coords[12][1])

            raised_hands_pose = raised_hands_raised_shoulders_ok and raised_hands_raised_hips_ok and raised_hands_hands_ok


            # --- Plank Pose Logic ---
            # Calculate angles between elbow-shoulder-hip for both sides
            
            plank_left_elbow_angle = compute_joint_angle_from_coords(coords, 11, 13, 15)
            plank_right_elbow_angle = compute_joint_angle_from_coords(coords, 12, 14, 16)

            plank_left_shoulder_angle = compute_joint_angle_from_coords(coords, 13,11,23)
            plank_right_shoulder_angle = compute_joint_angle_from_coords(coords, 14,12,24)

            plank_left_hip_angle = compute_joint_angle_from_coords(coords, 11,23,25)
            plank_right_hip_angle = compute_joint_angle_from_coords(coords, 12,24,26)

            plank_left_knee_angle = compute_joint_angle_from_coords(coords, 23,25,27)
            plank_right_knee_angle = compute_joint_angle_from_coords(coords, 24,26,28)

            plank_elbow_ok = (plank_left_elbow_angle >= 160) and (plank_right_elbow_angle >= 160)
            plank_shoulder_ok = (50 <= plank_left_shoulder_angle <= 100) and (50 <= plank_right_shoulder_angle <= 100)
            plank_hip_ok = (125 <=plank_left_hip_angle <= 195) and (125 <= plank_right_hip_angle <= 195)
            plank_knee_ok = (125 <= plank_left_knee_angle <= 195) and (125 <= plank_right_knee_angle <= 195)

            plank_pose = plank_elbow_ok and plank_shoulder_ok and plank_hip_ok and plank_knee_ok

            # Triangle Pose
            triangle_left_elbow = compute_joint_angle_from_coords(coords, 11, 13, 15)  # Left elbow angle
            triangle_right_elbow = compute_joint_angle_from_coords(coords, 12, 14, 16)  # Right elbow angle

            triangle_left_shoulder = compute_joint_angle_from_coords(coords, 13, 11, 23)  # Left shoulder-hip angle
            triangle_right_shoulder = compute_joint_angle_from_coords(coords, 14, 12, 24)  # Right shoulder-hip angle

            triangle_left_hip = compute_joint_angle_from_coords(coords, 11, 23, 25)  # Left hip angle
            triangle_right_hip = compute_joint_angle_from_coords(coords, 12, 24, 26)  # Right hip angle

            triangle_left_knee = compute_joint_angle_from_coords(coords, 23, 25, 27)  # Left knee angle
            triangle_right_knee = compute_joint_angle_from_coords(coords, 24, 26, 28)  # Right knee angle

            triangle_elbows_ok = triangle_left_elbow >= 160 and triangle_right_elbow >= 160
            triangle_shoulders_ok = (70 <= triangle_left_shoulder <= 100 and 100 <= triangle_right_shoulder <= 160) or \
                                    (70 <= triangle_right_shoulder <= 100 and 100 <= triangle_left_shoulder <= 160)
            triangle_hips_ok = (130 <= triangle_left_hip <= 160 and 50 <= triangle_right_hip <= 90) or \
                            (130 <= triangle_right_hip <= 160 and 50 <= triangle_left_hip <= 90)
            triangle_knees_ok = triangle_left_knee >= 160 and triangle_right_knee >= 160

            triangle_pose = triangle_elbows_ok and triangle_shoulders_ok and triangle_hips_ok and triangle_knees_ok


            # Extract landmarks for ML model.
            landmarks_ml = extract_landmarks(results)
            features = np.array(landmarks_ml).reshape(1, -1)

            if np.count_nonzero(features) == 0:
                pred_class = None
                max_prob = 0.0
            else:
                probabilities = model.predict_proba(features)
                max_prob = np.max(probabilities)
                if max_prob >= 0.7:
                    pred_class = model.classes_[np.argmax(probabilities)]
                else:
                    pred_class = None

            # Use if-elif-else chain for consistent logic.
            if pred_class == "Warrior Pose" and warrior_pose:
                pose_text = "Warrior Pose Detected (Correct)"
                skeleton_color = (0, 255, 0)
                accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
            elif pred_class == "Raised Hands" and raised_hands_pose:
                pose_text = "Raised Hands Pose Detected (Correct)"
                skeleton_color = (0, 255, 0)
                accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
            elif pred_class == "Plank Pose" and plank_pose:
                pose_text = "Plank Pose Detected (Correct)"
                skeleton_color = (0, 255, 0)
                accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
            elif pred_class == triangle_pose:
                pose_text = "Triangle Pose Detected (Correct)"
                skeleton_color = (0, 255, 0)
                accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
            else:
                pose_text = "Incorrect Pose - Adjust Your Position"
                skeleton_color = (0, 0, 255)

            draw_skeleton(display_frame, landmarks, mp_pose.POSE_CONNECTIONS, skeleton_color, 8, alpha=0.3)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        if accuracy_text:
            cv2.putText(display_frame, accuracy_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Real Time Universal Pose Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()