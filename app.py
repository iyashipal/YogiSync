from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import time
import pickle
import warnings
import base64
import os
from dotenv import load_dotenv
import google.generativeai as genai
import threading
import httpx  # (Not used in the LLM endpoint, but kept for your video/pose code)
import atexit
import signal

# Declare the global TTS thread variable.
tts_thread = None

def cleanup_threads():
    """Ensures all threads are stopped and subprocesses are cleaned up before program exits"""
    global tts_thread
    if 'tts_thread' in globals() and tts_thread is not None and tts_thread.is_alive():
        tts_thread.do_run = False  # Stop the loop inside the thread
        os.kill(os.getpid(), signal.SIGTERM)  # Terminate any child processes (including 'say')
        tts_thread.join(timeout=1)  # Ensure it exits

atexit.register(cleanup_threads)  # Register cleanup function

warnings.filterwarnings('ignore')

# Load environment variables (GOOGLE_API_KEY is needed for Gemini 1.5 Flash)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Optionally, you can specify a static folder for assets
app = Flask(__name__, static_folder='assets')

# ---------------------------
# Global variables and model
# ---------------------------
pose_info = {
    "pose_text": "No Pose Detected",
    "accuracy_text": "",
    "fps": 0.0,
    "correct": False,
    "llm_feedback": ""
}

workout_log = []
previous_report = {}
prev_pose_state = None
analysis_triggered_for_wrong = False
wrong_pose_start_time = None
DEBOUNCE_TIME = 1.5

# Load your pose classification model. Adjust the file path as needed.
model_path = os.getenv("MODEL_PATH", "yoga_v1.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
GAMMA = 1.2
invGamma = 1.0 / GAMMA
gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
PROCESS_WIDTH, PROCESS_HEIGHT = 640, 480
prev_time = 0
smoothed_landmarks = None
smoothing_factor = 0.7

# Global VideoCapture object and streaming flag.
camera = None
streaming_active = False

# ---------------------------
# Utility Functions
# ---------------------------
def preprocess_frame(frame):
    """Enhance the frame using CLAHE and gamma correction."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = CLAHE.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    final = cv2.LUT(enhanced, gamma_table)
    return final

def compute_joint_angle_from_coords(landmarks, i, j, k):
    """Compute the angle (in degrees) at landmark j using points i, j, k."""
    a = landmarks[i]
    b = landmarks[j]
    c = landmarks[k]
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def draw_skeleton(frame, landmarks, connections, color, thickness, alpha=0.6):
    """Draw the skeleton (only lines) on the frame."""
    overlay = frame.copy()
    body_connections = [
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28)   # Right leg
    ]
    for connection in body_connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx].x * frame.shape[1]),
                       int(landmarks[start_idx].y * frame.shape[0]))
        end_point = (int(landmarks[end_idx].x * frame.shape[1]),
                     int(landmarks[end_idx].y * frame.shape[0]))
        cv2.line(overlay, start_point, end_point, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def extract_landmarks(results):
    """Extract pose landmarks and return a flattened list."""
    row = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        row.extend([0] * (33 * 4))
    return row

def smooth_landmarks(new_landmarks):
    """
    Apply an exponential moving average to smooth landmark coordinates.
    Returns a numpy array with smoothed values.
    """
    global smoothed_landmarks, smoothing_factor
    new_coords = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in new_landmarks])
    if smoothed_landmarks is None:
        smoothed_landmarks = new_coords
    else:
        smoothed_landmarks = smoothing_factor * smoothed_landmarks + (1 - smoothing_factor) * new_coords
    return smoothed_landmarks

def ensure_camera():
    """Ensure the camera is initialized and opened."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

# ---------------------------
# Frame Generator Function
# ---------------------------
def gen_frames(selected_pose):
    """
    Generator function that captures frames from the camera, processes them
    with pose detection, applies smoothing, and yields JPEG-encoded images.
    The selected_pose parameter (e.g., 'warrior', 'raised', 'plank', 'triangle', 'chair', or 'tree')
    is used to determine which pose logic to apply.
    """
    global prev_time, pose_info, streaming_active, camera, tts_thread
    while streaming_active:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        display_frame = frame.copy()

        proc_frame = preprocess_frame(frame)
        proc_frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        proc_frame_rgb.flags.writeable = False

        results = pose.process(proc_frame_rgb)

        skeleton_color = (0, 0, 255)  # Default red color for incorrect pose.
        pose_text = "No Pose Detected"
        accuracy_text = ""
        correct = False

        if results.pose_landmarks:
            smoothed = smooth_landmarks(results.pose_landmarks.landmark)
            for i, lm in enumerate(results.pose_landmarks.landmark):
                lm.x, lm.y, lm.z, lm.visibility = smoothed[i]

            landmarks = results.pose_landmarks.landmark
            coords = np.array([[lm.x * PROCESS_WIDTH, lm.y * PROCESS_HEIGHT] for lm in landmarks])

            # --- Warrior Pose Logic ---
            left_elbow_angle = compute_joint_angle_from_coords(coords, 11, 13, 15)
            right_elbow_angle = compute_joint_angle_from_coords(coords, 12, 14, 16)
            left_knee_angle = compute_joint_angle_from_coords(coords, 23, 25, 27)
            right_knee_angle = compute_joint_angle_from_coords(coords, 24, 26, 28)
            left_arm_angle = compute_joint_angle_from_coords(coords, 23, 11, 13)
            right_arm_angle = compute_joint_angle_from_coords(coords, 24, 12, 14)
            warrior_elbows_ok = (150 <= left_elbow_angle <= 190) and (150 <= right_elbow_angle <= 190)
            warrior_knees_ok = (((110 <= left_knee_angle <= 150) and (150 <= right_knee_angle <= 190)) or
                                ((110 <= right_knee_angle <= 150) and (150 <= left_knee_angle <= 190)))
            arms_alignment_ok = (80 <= left_arm_angle <= 100) and (80 <= right_arm_angle <= 100)
            warrior_pose = warrior_elbows_ok and warrior_knees_ok and arms_alignment_ok

            # --- Raised Hands Pose Logic ---
            left_shoulder_angle = compute_joint_angle_from_coords(coords, 23, 11, 13)
            right_shoulder_angle = compute_joint_angle_from_coords(coords, 24, 12, 14)
            left_hip_angle = compute_joint_angle_from_coords(coords, 25, 23, 11)
            right_hip_angle = compute_joint_angle_from_coords(coords, 26, 24, 12)
            raised_shoulders_ok = (160 <= left_shoulder_angle <= 180) and (160 <= right_shoulder_angle <= 180)
            raised_hips_ok = (left_hip_angle < 180) and (right_hip_angle < 180)
            raised_hands_ok = (coords[15][1] < coords[11][1]) and (coords[16][1] < coords[12][1])
            raised_hands_pose = raised_shoulders_ok and raised_hips_ok and raised_hands_ok

            # --- Plank Pose Logic ---
            plank_left_elbow_angle = compute_joint_angle_from_coords(coords, 11, 13, 15)
            plank_right_elbow_angle = compute_joint_angle_from_coords(coords, 12, 14, 16)
            plank_left_shoulder_angle = compute_joint_angle_from_coords(coords, 13, 11, 23)
            plank_right_shoulder_angle = compute_joint_angle_from_coords(coords, 14, 12, 24)
            plank_left_hip_angle = compute_joint_angle_from_coords(coords, 11, 23, 25)
            plank_right_hip_angle = compute_joint_angle_from_coords(coords, 12, 24, 26)
            plank_left_knee_angle = compute_joint_angle_from_coords(coords, 23, 25, 27)
            plank_right_knee_angle = compute_joint_angle_from_coords(coords, 24, 26, 28)
            plank_elbow_ok = (plank_left_elbow_angle >= 160) and (plank_right_elbow_angle >= 160)
            plank_shoulder_ok = (50 <= plank_left_shoulder_angle <= 100) and (50 <= plank_right_shoulder_angle <= 100)
            plank_hip_ok = (125 <= plank_left_hip_angle <= 195) and (125 <= plank_right_hip_angle <= 195)
            plank_knee_ok = (125 <= plank_left_knee_angle <= 195) and (125 <= plank_right_knee_angle <= 195)
            plank_pose = plank_elbow_ok and plank_shoulder_ok and plank_hip_ok and plank_knee_ok

            # --- Triangle Pose Logic ---
            triangle_left_elbow = compute_joint_angle_from_coords(coords, 11, 13, 15)
            triangle_right_elbow = compute_joint_angle_from_coords(coords, 12, 14, 16)
            triangle_left_shoulder = compute_joint_angle_from_coords(coords, 13, 11, 23)
            triangle_right_shoulder = compute_joint_angle_from_coords(coords, 14, 12, 24)
            triangle_left_hip = compute_joint_angle_from_coords(coords, 11, 23, 25)
            triangle_right_hip = compute_joint_angle_from_coords(coords, 12, 24, 26)
            triangle_left_knee = compute_joint_angle_from_coords(coords, 23, 25, 27)
            triangle_right_knee = compute_joint_angle_from_coords(coords, 24, 26, 28)
            triangle_elbows_ok = (triangle_left_elbow >= 160) and (triangle_right_elbow >= 160)
            triangle_shoulders_ok = ((70 <= triangle_left_shoulder <= 100 and 100 <= triangle_right_shoulder <= 160) or
                                     (70 <= triangle_right_shoulder <= 100 and 100 <= triangle_left_shoulder <= 160))
            triangle_hips_ok = ((130 <= triangle_left_hip <= 160 and 50 <= triangle_right_hip <= 90) or
                                (130 <= triangle_right_hip <= 160 and 50 <= triangle_left_hip <= 90))
            triangle_knees_ok = (triangle_left_knee >= 160) and (triangle_right_knee >= 160)
            triangle_pose = triangle_elbows_ok and triangle_shoulders_ok and triangle_hips_ok and triangle_knees_ok

            # --- Chair Pose Logic ---
            chair_left_elbow = compute_joint_angle_from_coords(coords, 11, 13, 15)
            chair_right_elbow = compute_joint_angle_from_coords(coords, 12, 14, 16)
            chair_left_shoulder = compute_joint_angle_from_coords(coords, 13, 11, 23)
            chair_right_shoulder = compute_joint_angle_from_coords(coords, 14, 12, 24)
            chair_left_hip = compute_joint_angle_from_coords(coords, 11, 23, 25)
            chair_right_hip = compute_joint_angle_from_coords(coords, 12, 24, 26)
            chair_left_knee = compute_joint_angle_from_coords(coords, 23, 25, 27)
            chair_right_knee = compute_joint_angle_from_coords(coords, 24, 26, 28)
            chair_left_elbow_ok = (155 <= chair_left_elbow <= 190)
            chair_right_elbow_ok = (155 <= chair_right_elbow <= 190)
            chair_left_hip_ok = (110 <= chair_left_hip <= 160)
            chair_right_hip_ok = (110 <= chair_right_hip <= 160)
            chair_left_knee_ok = (110 <= chair_left_knee <= 160)
            chair_right_knee_ok = (110 <= chair_right_knee <= 160)
            chair_left_shoulder_ok = (130 <= chair_left_shoulder <= 180)
            chair_right_shoulder_ok = (130 <= chair_right_shoulder <= 180)
            chair_pose = (chair_left_elbow_ok and chair_right_elbow_ok) and (chair_left_hip_ok and chair_right_hip_ok) and (chair_left_knee_ok and chair_right_knee_ok) and (chair_left_shoulder_ok and chair_right_shoulder_ok)

            # --- Tree Pose Logic ---
            tree_left_elbow = compute_joint_angle_from_coords(coords, 11, 13, 15)
            tree_right_elbow = compute_joint_angle_from_coords(coords, 12, 14, 16)
            tree_left_shoulder = compute_joint_angle_from_coords(coords, 13, 11, 23)
            tree_right_shoulder = compute_joint_angle_from_coords(coords, 14, 12, 24)
            tree_left_hip = compute_joint_angle_from_coords(coords, 11, 23, 25)
            tree_right_hip = compute_joint_angle_from_coords(coords, 12, 24, 26)
            tree_left_knee = compute_joint_angle_from_coords(coords, 23, 25, 27)
            tree_right_knee = compute_joint_angle_from_coords(coords, 24, 26, 28)
            tree_elbows_ok = (130 <= tree_left_elbow <= 180) and (130 <= tree_right_elbow <= 180)
            tree_shoulders_ok = (130 <= tree_left_shoulder <= 180) and (130 <= tree_right_shoulder <= 180)
            tree_hips_ok = ((90 <= tree_left_hip <= 160 and 150 <= tree_right_hip <= 180) or
                            (90 <= tree_right_hip <= 155 and 155 <= tree_left_hip <= 180))
            tree_knees_ok = ((0 <= tree_left_knee <= 90 and 150 <= tree_right_knee <= 180) or
                             (0 <= tree_right_knee <= 90 and 150 <= tree_left_knee <= 180))
            tree_pose = tree_elbows_ok and tree_shoulders_ok and tree_hips_ok and tree_knees_ok

            # ---------------------------
            # Pose Classification via ML Model
            # ---------------------------
            landmarks_ml = extract_landmarks(results)
            features = np.array(landmarks_ml).reshape(1, -1)

            if np.count_nonzero(features) == 0:
                predicted_text = "No Pose Detected"
            else:
                probabilities = model.predict_proba(features)
                max_prob = np.max(probabilities)
                pred_class = model.classes_[np.argmax(probabilities)]
                
                # --- Determine final output based on selected pose ---
                if selected_pose == "warrior":
                    if pred_class == "Warrior Pose" and warrior_pose:
                        pose_text = "Warrior Pose Detected (Correct)"
                        correct = True
                        skeleton_color = (0, 255, 0)
                        accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
                        if tts_thread is not None and tts_thread.is_alive():
                            tts_thread.do_run = False
                            tts_thread = None
                    else:
                        pose_text = "Incorrect Warrior Pose - Adjust Your Position"
                        correct = False
                        skeleton_color = (0, 0, 255)
                        if pose_info["correct"] and (tts_thread is None or not tts_thread.is_alive()):
                            def tts():
                                t = threading.current_thread()
                                while getattr(t, "do_run", True):
                                    for sentence in instructions.split('\n'):
                                        if sentence.strip():
                                            os.system(f'say "{sentence.strip()}"')
                                            time.sleep(2)
                                            if not getattr(t, "do_run", True):
                                                return
                                    time.sleep(3)
                            tts_thread = threading.Thread(target=tts, daemon=True)
                            tts_thread.start()

                elif selected_pose == "raised":
                    if raised_hands_pose:
                        pose_text = "Raised Hands Pose Detected (Correct)"
                        correct = True
                        skeleton_color = (0, 255, 0)
                        accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
                        if tts_thread is not None and tts_thread.is_alive():
                            tts_thread.do_run = False
                            os.system("pkill -f say")
                            tts_thread = None
                    else:
                        pose_text = "Incorrect Raised Hands Pose - Adjust Your Position"
                        correct = False
                        skeleton_color = (0, 0, 255)
                        if pose_info["correct"] and (tts_thread is None or not tts_thread.is_alive()):
                            def tts():
                                t = threading.current_thread()
                                while getattr(t, "do_run", True):
                                    for sentence in instructions.split('\n'):
                                        if sentence.strip():
                                            os.system(f'say "{sentence.strip()}"')
                                            time.sleep(2)
                                            if not getattr(t, "do_run", True):
                                                return
                                    time.sleep(3)
                            tts_thread = threading.Thread(target=tts, daemon=True)
                            tts_thread.start()

                elif selected_pose == "plank":
                    if pred_class == "Plank Pose" and plank_pose:
                        pose_text = "Plank Pose Detected (Correct)"
                        correct = True
                        skeleton_color = (0, 255, 0)
                        accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
                        if tts_thread is not None and tts_thread.is_alive():
                            tts_thread.do_run = False
                            os.system("pkill -f say")
                            tts_thread = None
                    else:
                        pose_text = "Incorrect Plank Pose - Adjust Your Position"
                        correct = False
                        skeleton_color = (0, 0, 255)
                        if pose_info["correct"] and (tts_thread is None or not tts_thread.is_alive()):
                            def tts():
                                t = threading.current_thread()
                                while getattr(t, "do_run", True):
                                    for sentence in instructions.split('\n'):
                                        if sentence.strip():
                                            os.system(f'say "{sentence.strip()}"')
                                            time.sleep(2)
                                            if not getattr(t, "do_run", True):
                                                return
                                    time.sleep(3)
                            tts_thread = threading.Thread(target=tts, daemon=True)
                            tts_thread.start()

                elif selected_pose == "triangle":
                    if pred_class == "Triangle Pose" and triangle_pose:
                        pose_text = "Triangle Pose Detected (Correct)"
                        correct = True
                        skeleton_color = (0, 255, 0)
                        accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
                        if tts_thread is not None and tts_thread.is_alive():
                            tts_thread.do_run = False
                            os.system("pkill -f say")
                            tts_thread = None
                    else:
                        pose_text = "Incorrect Triangle Pose - Adjust Your Position"
                        correct = False
                        skeleton_color = (0, 0, 255)
                        if pose_info["correct"] and (tts_thread is None or not tts_thread.is_alive()):
                            def tts():
                                t = threading.current_thread()
                                while getattr(t, "do_run", True):
                                    for sentence in instructions.split('\n'):
                                        if sentence.strip():
                                            os.system(f'say "{sentence.strip()}"')
                                            time.sleep(2)
                                            if not getattr(t, "do_run", True):
                                                return
                                    time.sleep(3)
                            tts_thread = threading.Thread(target=tts, daemon=True)
                            tts_thread.start()

                elif selected_pose == "chair":
                    if pred_class == "Chair Pose" and chair_pose:
                        pose_text = "Chair Pose Detected (Correct)"
                        correct = True
                        skeleton_color = (0, 255, 0)
                        accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
                        if tts_thread is not None and tts_thread.is_alive():
                            tts_thread.do_run = False
                            os.system("pkill -f say")
                            tts_thread = None
                    else:
                        pose_text = "Incorrect Chair Pose - Adjust Your Position"
                        correct = False
                        skeleton_color = (0, 0, 255)
                        if pose_info["correct"] and (tts_thread is None or not tts_thread.is_alive()):
                            def tts():
                                t = threading.current_thread()
                                while getattr(t, "do_run", True):
                                    for sentence in instructions.split('\n'):
                                        if sentence.strip():
                                            os.system(f'say "{sentence.strip()}"')
                                            time.sleep(2)
                                            if not getattr(t, "do_run", True):
                                                return
                                    time.sleep(3)
                            tts_thread = threading.Thread(target=tts, daemon=True)
                            tts_thread.start()

                elif selected_pose == "tree":
                    if pred_class == "Tree Pose" and tree_pose:
                        pose_text = "Tree Pose Detected (Correct)"
                        correct = True
                        skeleton_color = (0, 255, 0)
                        accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
                        if tts_thread is not None and tts_thread.is_alive():
                            tts_thread.do_run = False
                            os.system("pkill -f say")
                            tts_thread = None
                    else:
                        pose_text = "Incorrect Tree Pose - Adjust Your Position"
                        correct = False
                        skeleton_color = (0, 0, 255)
                        if pose_info["correct"] and (tts_thread is None or not tts_thread.is_alive()):
                            def tts():
                                t = threading.current_thread()
                                while getattr(t, "do_run", True):
                                    for sentence in instructions.split('\n'):
                                        if sentence.strip():
                                            os.system(f'say "{sentence.strip()}"')
                                            time.sleep(2)
                                            if not getattr(t, "do_run", True):
                                                return
                                    time.sleep(3)
                            tts_thread = threading.Thread(target=tts, daemon=True)
                            tts_thread.start()

            if not streaming_active and tts_thread is not None and tts_thread.is_alive():
                tts_thread.do_run = False
                os.system("pkill -f say")
                tts_thread = None

            draw_skeleton(display_frame, landmarks, mp_pose.POSE_CONNECTIONS, skeleton_color, 8, alpha=0.3)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        pose_info["pose_text"] = pose_text
        pose_info["accuracy_text"] = accuracy_text
        pose_info["fps"] = fps
        pose_info["correct"] = correct

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------------------
# Original Endpoints
# ---------------------------
@app.route('/')
def index():
    """Homepage with Practice button."""
    return render_template('index.html')

@app.route('/practice')
def practice():
    """Pose selection page."""
    return render_template('practice.html')

@app.route('/video/<pose_choice>')
def video(pose_choice):
    """
    Practice page for the selected pose.
    This page will be split into two columns: instructions and camera feed.
    """
    return render_template('video.html', pose_choice=pose_choice)

@app.route('/video_feed/<pose_choice>')
def video_feed(pose_choice):
    """Video stream endpoint."""
    global streaming_active
    ensure_camera()
    streaming_active = True
    return Response(gen_frames(pose_choice), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop the video stream and release the camera."""
    global streaming_active, camera, smoothed_landmarks
    streaming_active = False
    smoothed_landmarks = None
    if camera is not None and camera.isOpened():
        camera.release()
    return jsonify({"status": "stopped"})

@app.route('/pose_info')
def get_pose_info():
    """Return the current pose info as JSON."""
    return jsonify(pose_info)

@app.route('/learn/<pose_choice>')
def learn(pose_choice):
    """Endpoint to render the learn page for a given pose."""
    return render_template('learn.html', pose_choice=pose_choice)

# ---------------------------
# Chat and Playlist Endpoints
# ---------------------------
playlist = []

@app.route('/api/add_to_playlist', methods=['POST'])
def add_to_playlist():
    data = request.get_json()
    pose = data.get("pose", "")
    if pose:
        playlist.append(pose)
        return jsonify({"status": "added"})
    else:
        return jsonify({"status": "error", "message": "No pose provided"}), 400

@app.route('/playlist')
def show_playlist():
    return render_template('playlist.html', playlist=playlist)

@app.route('/api/update_playlist_order', methods=['POST'])
def update_playlist_order():
    data = request.get_json()
    new_order = data.get('order', [])
    if new_order:
        global playlist
        playlist = new_order
        return jsonify({"status": "updated", "playlist": playlist})
    return jsonify({"status": "error", "message": "No order provided"}), 400

@app.route('/start_practice/<int:pose_index>')
def start_practice(pose_index):
    if pose_index < len(playlist):
        pose_name = playlist[pose_index]
        mapping = {
            "Warrior Pose": "warrior",
            "Raised Hands Pose": "raised",
            "Plank Pose": "plank",
            "Triangle Pose": "triangle",
            "Chair Pose": "chair",
            "Tree Pose": "tree"
        }
        pose_choice = mapping.get(pose_name, pose_name.lower())
        return render_template('video.html', 
                               pose_choice=pose_choice, 
                               pose_index=pose_index, 
                               playlist=playlist)
    else:
        return redirect(url_for('practice'))

@app.route('/stop_tts', methods=['POST'])
def stop_tts():
    global tts_thread
    if tts_thread is not None and tts_thread.is_alive():
        tts_thread.do_run = False
        os.system("pkill -f say")
        tts_thread = None
    return jsonify({"status": "tts stopped"})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_message = data.get("message", "")
    
    # Simple greeting response.
    if user_message.strip().lower() in ["hi", "hello", "hey"]:
        return jsonify({"reply": "Hi there! How can I help you with your yoga practice today?"})
    
    prompt_text = (
        "You are a professional and emotionally intelligent yoga instructor. Please ensure that your responses are concise, helpful, and visually neat.\n\n"
        "Guidelines:\n"
        "1. If the user greets you (e.g., 'hi', 'hello', 'hey'), respond with a brief greeting.\n\n"
        "2. If the user describes a body issue or problem, choose exactly one pose from this approved list that would help them: Warrior Pose, Raised Hands Pose, Triangle Pose, Tree Pose, Pank Pose.\n"
        "   • Present the pose name in **bold** (e.g., **Warrior Pose**).\n"
        "   • Provide a single short sentence explaining why it helps.\n"
        "   • Then recommend one brief mindfulness or breathing exercise in another short sentence.\n"
        "   • Format each point as a bullet, using '•' or '-'.\n"
        "   • End with '[Add to Playlist]' on its own line.\n\n"
        "3. If the user says 'thank you', 'good night', or 'bye', reply with a concise farewell.\n\n"
        "4. For other queries, respond briefly but helpfully.\n\n"
        "User says: '{user_message}'\n"
        "Now produce your final response accordingly."
    )
    prompt_text = prompt_text.replace("{user_message}", user_message)
    
    payload = {"parts": [{"text": prompt_text}]}
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = llm_model.generate_content(payload)
        reply = response.text if hasattr(response, "text") else "I'm sorry, I didn't understand that."
    except Exception as e:
        print("LLM error:", e)
        reply = "I'm sorry, I'm having trouble responding right now."
    
    return jsonify({"reply": reply})

@app.route('/chat')
def chat():
    """
    Renders chat.html, which contains a full-screen popup questionnaire.
    Once the user completes the questions, it calls /generate_routine
    and displays the plan on the same page.
    """
    return render_template('chat.html')

@app.route('/generate_routine', methods=['POST'])
def generate_routine():
    data = request.get_json()

    # Construct a prompt that requests HTML formatting and includes all new data
    prompt = (
        "Please create a professional, HTML-formatted yoga & fitness plan report. "
        "Use headings (<h3>) and bullet points (<ul><li>) for clarity. "
        "Main Heading must be Client Summary"
        "Make sure the report includes:\n\n"
        "1. A short summary of the user's information (age, gender, height, weight, etc.).\n"
        "2. A detailed diet plan with approximate calorie consumption, recommended macronutrient ratios, "
        "   and suggested caloric intake based on the user's goals.\n"
        "3. A workout routine laid out per day (like Day 1, Day 2...), with recommended yoga poses or exercises.\n"
        "4. A conclusion discussing how long it might take to reach their set goal.\n\n"
        "Here is the user data:\n"
        f"- Age Range: {data.get('age_range')}\n"
        f"- Gender: {data.get('gender')}\n"
        f"- Height: {data.get('height')}\n"
        f"- Weight: {data.get('weight')}\n"
        f"- Primary Goal: {data.get('primary_goal')}\n"
        f"- Fitness Level: {data.get('fitness_level')}\n"
        f"- Activity Level: {data.get('activity_level')}\n"
        f"- Current Eating Habits: {data.get('current_eating_habits')}\n"
        f"- Eating Habit Goals: {data.get('eating_goals')}\n"
        f"- Dietary Preferences/Restrictions: {data.get('dietary_restrictions')}\n"
        f"- Weeks to Work on Routine: {data.get('weeks')}\n"
        f"- Warm-up Preference: {data.get('warmup')}\n"
        f"- Breathing Preference: {data.get('breathing')}\n"
        f"- Current Workout Status: {data.get('workout_status')}\n"
        f"- Desired Workout Duration: {data.get('workout_duration')}\n"
        f"- Health Concerns/Injuries: {data.get('health_concerns')}\n\n"
        "Include all relevant details in the final plan. Format everything nicely in HTML. Put client summary - Generate a concise and engaging summary of the user’s lifestyle and fitness journey in paragraph form (max 150 words). Avoid listing specific details; instead, infer what their information reveals about their habits, priorities, and challenges. The tone should be motivating yet neutral, acknowledging their current status while emphasizing their aspirations.- in paragraph."
        "do NOT print triple back ticks. NEVER PRINT THEM it should not print '''html"
    )
    
    try:
        llm_model = genai.GenerativeModel('gemini-1.5-flash')
        payload = {"parts": [{"text": prompt}]}
        response = llm_model.generate_content(payload)
        # If Gemini returns a text attribute, treat it as our HTML
        detailed_html = response.text if hasattr(response, "text") else "No HTML plan generated."
    except Exception as e:
        print("Gemini error:", e)
        detailed_html = (
            "<p>We're sorry, but there was an error generating your plan at this time.</p>"
        )

    return jsonify({"routine": detailed_html})

# ---------------------------
# End of Endpoints
# ---------------------------


# ---------------------------
# End of Endpoints
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
