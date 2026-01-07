# import cv2
# import csv
# import mediapipe as mp
# import numpy as np
#
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
#
#
# def calculateAngle(landmark1, landmark2, landmark3):
#     x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
#     x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
#     x3, y3, _ = landmark3.x, landmark3.y, landmark3.z
#
#     angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) -
#                        np.arctan2(y1 - y2, x1 - x2))
#     if angle < 0:
#         angle += 360
#     return angle
#
#
# def collect_pose_from_video(video_path, output_csv):
#     cap = cv2.VideoCapture(video_path)
#
#     # Prepare CSV
#     with open(output_csv, mode='w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow([
#             'left_wrist_angle', 'right_wrist_angle', 'left_elbow_angle', 'right_elbow_angle',
#             'left_shoulder_angle', 'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle',
#             'left_ankle_angle', 'right_ankle_angle', 'left_hip_angle', 'right_hip_angle', 'name_yoga'
#         ])
#
#         with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#
#                 image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = pose.process(image_rgb)
#
#                 # Draw landmarks on the frame for guidance
#                 if results.pose_landmarks:
#                     mp_drawing.draw_landmarks(
#                         frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
#                 cv2.imshow("Yoga Video - Press 'c' to capture pose, 'q' to quit", frame)
#
#                 key = cv2.waitKey(1) & 0xFF
#
#                 if key == ord('c') and results.pose_landmarks:
#                     landmarks = results.pose_landmarks.landmark
#                     angles = []
#
#                     # Collect 12 angles
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
#                                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]))
#
#                     angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]))
#
#                     # Ask user for yoga pose name
#                     pose_name = input("Enter yoga pose name for this frame: ")
#
#                     # Save to CSV
#                     csv_writer.writerow(angles + [pose_name])
#                     print(f"Saved {pose_name} with angles {angles}")
#
#                 elif key == ord('q'):
#                     break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Example usage
# video_file = "downdog_warrior2.mp4"
# output_csv = "yoga_pose_dataset.csv"
# collect_pose_from_video(video_file, output_csv)


import cv2
import csv
import mediapipe as mp
import numpy as np


class YogaPoseCollector:
    def __init__(self, video_path, output_csv,
                 min_detection_conf=0.5, min_tracking_conf=0.5):
        self.video_path = video_path
        self.output_csv = output_csv
        self.cap = cv2.VideoCapture(video_path)

        # Mediapipe pose and drawing utils
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )

        # Prepare CSV writer
        self.csvfile = open(self.output_csv, mode='w', newline='')
        self.csv_writer = csv.writer(self.csvfile)
        self.csv_writer.writerow([
            'left_wrist_angle', 'right_wrist_angle',
            'left_elbow_angle', 'right_elbow_angle',
            'left_shoulder_angle', 'right_shoulder_angle',
            'left_knee_angle', 'right_knee_angle',
            'left_ankle_angle', 'right_ankle_angle',
            'left_hip_angle', 'right_hip_angle',
            'name_yoga'
        ])

    def calculate_angle(self, landmark1, landmark2, landmark3):
        x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
        x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
        x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

        angle = np.degrees(
            np.arctan2(y3 - y2, x3 - x2) -
            np.arctan2(y1 - y2, x1 - x2)
        )
        if angle < 0:
            angle += 360
        return angle

    def extract_angles(self, landmarks):
        mp_pose = self.mp_pose
        angles = []

        # Wrists
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
            landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]))
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]))

        # Elbows
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]))
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]))

        # Shoulders
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]))
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]))

        # Knees
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))

        # Ankles
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]))
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]))

        # Hips
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]))
        angles.append(self.calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]))

        return angles

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Yoga Video - Press 'c' to capture pose, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                angles = self.extract_angles(landmarks)

                pose_name = input("Enter yoga pose name for this frame: ")
                self.csv_writer.writerow(angles + [pose_name])
                print(f"Saved {pose_name} with angles {angles}")

            elif key == ord('q'):
                break

        self.cap.release()
        self.csvfile.close()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    video_file = "downdog_warrior2.mp4"
    output_csv = "yoga_pose_dataset.csv"

    collector = YogaPoseCollector(video_file, output_csv)
    collector.run()
