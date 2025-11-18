import cv2
import mediapipe as mp

# Initialize MediaPipe drawing and pose modules.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize webcam capture.
cap = cv2.VideoCapture(0)

# Setup MediaPipe Pose.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR frame to RGB as MediaPipe requires RGB input.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Perform pose detection.
        results = pose.process(image)
        
        # Convert the image back to BGR for OpenCV.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks on the frame.
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=10, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=10, circle_radius=1))
        
        # Display the resulting frame.
        cv2.imshow('Mediapipe Pose Estimation', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
