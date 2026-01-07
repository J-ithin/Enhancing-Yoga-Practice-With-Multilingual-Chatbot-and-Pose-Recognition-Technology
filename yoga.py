import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

# Load average yoga feature values from dataset
data_path = 'yoga_pose_dataset.csv'
yoga_features = pd.read_csv(data_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to detect pose based on angles
def detect_pose(angles):
    min_diff = float('inf')
    detected_pose = None

    for _, row in yoga_features.iterrows():
        pose_name = row['name_yoga']
        total_diff = 0

        for key in angles.keys():
            avg_value = row[key]
            total_diff += abs(angles[key] - avg_value)

        if total_diff < min_diff:
            min_diff = total_diff
            detected_pose = pose_name

    return detected_pose


# OpenCV video capture
cap = cv2.VideoCapture('downdog_warrior2.mp4')
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame and get pose landmarks
        results = pose.process(image)

        # Convert image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Define key points for angle calculations
            keypoints = {
                "left_wrist": [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
                "right_wrist": [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y],
                "left_elbow": [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                "right_elbow": [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                "right_shoulder": [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                "left_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                "right_hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
                "left_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                "right_knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
                "left_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                "right_ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            }

            # Calculate angles
            angles = {
                "left_wrist_angle": calculate_angle(keypoints["left_elbow"], keypoints["left_wrist"],
                                                    keypoints["left_shoulder"]),
                "right_wrist_angle": calculate_angle(keypoints["right_elbow"], keypoints["right_wrist"],
                                                     keypoints["right_shoulder"]),
                "left_elbow_angle": calculate_angle(keypoints["left_wrist"], keypoints["left_elbow"],
                                                    keypoints["left_shoulder"]),
                "right_elbow_angle": calculate_angle(keypoints["right_wrist"], keypoints["right_elbow"],
                                                     keypoints["right_shoulder"]),
                "left_shoulder_angle": calculate_angle(keypoints["left_elbow"], keypoints["left_shoulder"],
                                                       keypoints["left_hip"]),
                "right_shoulder_angle": calculate_angle(keypoints["right_elbow"], keypoints["right_shoulder"],
                                                        keypoints["right_hip"]),
                "left_knee_angle": calculate_angle(keypoints["left_hip"], keypoints["left_knee"],
                                                   keypoints["left_ankle"]),
                "right_knee_angle": calculate_angle(keypoints["right_hip"], keypoints["right_knee"],
                                                    keypoints["right_ankle"]),
                "left_ankle_angle": calculate_angle(keypoints["left_knee"], keypoints["left_ankle"],
                                                    keypoints["left_wrist"]),
                "right_ankle_angle": calculate_angle(keypoints["right_knee"], keypoints["right_ankle"],
                                                     keypoints["right_wrist"]),
                "left_hip_angle": calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"],
                                                  keypoints["left_knee"]),
                "right_hip_angle": calculate_angle(keypoints["right_shoulder"], keypoints["right_hip"],
                                                   keypoints["right_knee"])
            }

            # Detect pose
            detected_pose = detect_pose(angles)

            # Provide feedback for corrections
            feedback = {}
            pose_row = yoga_features[yoga_features['name_yoga'] == detected_pose].iloc[0]
            for key in angles.keys():
                avg_value = pose_row[key]
                diff_percentage = abs((angles[key] - avg_value) / avg_value) * 100
                feedback[key] = diff_percentage

            # Display feedback and detected pose on the frame
            cv2.putText(image, f"Pose: {detected_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset = 50
            for key, diff in feedback.items():
                cv2.putText(image, f"{key}: {diff:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                y_offset += 20

            # Draw the pose on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the frame
        cv2.imshow('Yoga Pose Detection and Correction', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
