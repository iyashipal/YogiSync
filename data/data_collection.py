import os
import mediapipe as mp
import cv2
import numpy as np
import csv

# Initialize MediaPipe drawing and pose modules.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def preprocess_frame(frame, gamma=1.2):

    # Convert the frame to LAB color space.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with a and b channels.
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply gamma correction to adjust brightness.
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    final = cv2.LUT(enhanced, table)
    
    return final

def write_csv_header(csv_filename, num_pose_landmarks, include_face_landmarks=False):
 
    header = ['class']
    for i in range(1, num_pose_landmarks + 1):
        header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
    
    if include_face_landmarks:
        # Assuming 468 face landmarks (typical for face mesh).
        num_face_landmarks = 468
        for i in range(1, num_face_landmarks + 1):
            header.extend([f'fx{i}', f'fy{i}', f'fz{i}', f'fv{i}'])
    
    with open(csv_filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)

def extract_landmarks(results, include_face=False):
   
    row = []
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        for landmark in pose_landmarks:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        # If no landmarks are detected, fill with zeros for all pose landmarks.
        row.extend([0] * (33 * 4))  # 33 landmarks x 4 values each
    
    if include_face:
        if results.face_landmarks:
            face_landmarks = results.face_landmarks.landmark
            for landmark in face_landmarks:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            # Fill with zeros for 468 face landmarks if none are detected.
            row.extend([0] * (468 * 4))
    return row

def main():
    
    csv_filename = r'D:\YogiSync\data\dataset\landmarks.csv'
    # Change class_name as needed for each pose (e.g., "warrior_pose", "tree_pose", etc.)
    class_name = "Triangle Pose"
    num_pose_landmarks = 33
    include_face_landmarks = False  # Set True if you want to log face landmarks too.
    
    # Check if the CSV file already exists; if not, write the header.
    if not os.path.exists(csv_filename):
        write_csv_header(csv_filename, num_pose_landmarks, include_face_landmarks)
    
    # Initialize video capture.
    cap = cv2.VideoCapture(0)
    
    # Setup MediaPipe Pose.
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save the original frame for live display.
            orig_frame = frame.copy()
            
            # Preprocess a copy of the frame (for improved detection).
            proc_frame = preprocess_frame(frame, gamma=1.2)
            
            # Convert the preprocessed frame to RGB (MediaPipe expects RGB).
            proc_frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            proc_frame_rgb.flags.writeable = False
            
            # Perform pose detection on the enhanced frame.
            results = pose.process(proc_frame_rgb)
            
            # Draw landmarks on the original frame (live feed remains unaltered).
            mp_drawing.draw_landmarks(orig_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=10, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=10, circle_radius=1))
            
            # Extract landmarks and prepare a CSV row.
            row = [class_name]
            row.extend(extract_landmarks(results, include_face=include_face_landmarks))
            
            # Append the row to the CSV file.
            with open(csv_filename, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
            # Show the live feed with overlaid landmarks.
            cv2.imshow('Mediapipe Feed', orig_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
