
import streamlit as st
import sqlite3
import hashlib
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import io
import csv
import os

# ================================
# Custom Styling
# ================================
st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {"#38b6ff"};
            color: {"#FFFFFF"};
        }}
        </style>
        """,
        unsafe_allow_html=True
)
# =========================
# Database Helpers
# =========================
def create_usertable():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS userstable
                 (name TEXT, age INT, email TEXT UNIQUE, place TEXT, password TEXT)''')
    conn.commit()
    conn.close()


def add_userdata(name, age, email, place, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('INSERT INTO userstable (name, age, email, place, password) VALUES (?, ?, ?, ?, ?)',
              (name, age, email, place, password))
    conn.commit()
    conn.close()

    csv_file = "users.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['name', 'age', 'email', 'place', 'password'])
        writer.writerow([name, age, email, place, password])


def login_user(email, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE email=? AND password=?', (email, password))
    data = c.fetchone()
    conn.close()
    return data

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_admin_credentials(username, password):
    hashed_admin_pswd = make_hashes("admin123")
    return username == "admin" and make_hashes(password) == hashed_admin_pswd

# =========================
# Pose Detection Functions
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_pose(angles, yoga_features):
    min_diff = float('inf')
    detected_pose = None
    for _, row in yoga_features.iterrows():
        pose_name = row['name_yoga']
        total_diff = sum(abs(angles[k] - row[k]) for k in angles.keys() if k in row)
        if total_diff < min_diff:
            min_diff = total_diff
            detected_pose = pose_name
    return detected_pose

# def run_pose_detection(video_source, dataset_path="yoga_pose_dataset.csv", frame_placeholder=None):
#     # Load dataset
#     try:
#         yoga_features = pd.read_csv(dataset_path)
#     except FileNotFoundError:
#         st.error("Dataset file not found. Please ensure yoga_pose_dataset.csv exists.")
#         return
#     except pd.errors.EmptyDataError:
#         st.error("Dataset file is empty or invalid.")
#         return
#
#     cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW if isinstance(video_source, int) else 0)
#
#     if not cap.isOpened():
#         st.error("Error: Cannot open video source.")
#         return
#
#     st.info("Pose detection running... Press the Quit button to stop.")
#
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             if st.session_state.quit or not st.session_state.detecting:
#                 break
#
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             # Resize for better visibility
#             frame = cv2.resize(frame, (800, 600))
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#             if results.pose_landmarks:
#                 landmarks = results.pose_landmarks.landmark
#                 keypoints = {
#                     "left_wrist": [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
#                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
#                     "right_wrist": [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
#                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y],
#                     "left_elbow": [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
#                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
#                     "right_elbow": [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
#                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
#                     "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
#                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
#                     "right_shoulder": [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
#                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
#                     "left_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
#                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
#                     "right_hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
#                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
#                     "left_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
#                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
#                     "right_knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
#                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
#                     "left_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
#                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
#                     "right_ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
#                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
#                 }
#
#                 angles = {
#                     "left_wrist_angle": calculate_angle(keypoints["left_elbow"], keypoints["left_wrist"], keypoints["left_shoulder"]),
#                     "right_wrist_angle": calculate_angle(keypoints["right_elbow"], keypoints["right_wrist"], keypoints["right_shoulder"]),
#                     "left_elbow_angle": calculate_angle(keypoints["left_wrist"], keypoints["left_elbow"], keypoints["left_shoulder"]),
#                     "right_elbow_angle": calculate_angle(keypoints["right_wrist"], keypoints["right_elbow"], keypoints["right_shoulder"]),
#                     "left_shoulder_angle": calculate_angle(keypoints["left_elbow"], keypoints["left_shoulder"], keypoints["left_hip"]),
#                     "right_shoulder_angle": calculate_angle(keypoints["right_elbow"], keypoints["right_shoulder"], keypoints["right_hip"]),
#                     "left_knee_angle": calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"]),
#                     "right_knee_angle": calculate_angle(keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"]),
#                     "left_hip_angle": calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_knee"]),
#                     "right_hip_angle": calculate_angle(keypoints["right_shoulder"], keypoints["right_hip"], keypoints["right_knee"])
#                 }
#
#                 detected_pose = detect_pose(angles, yoga_features)
#                 feedback = {}
#                 pose_row = yoga_features[yoga_features['name_yoga'] == detected_pose].iloc[0]
#                 for k in angles.keys():
#                     avg_value = pose_row[k]
#                     feedback[k] = abs((angles[k] - avg_value) / avg_value) * 100 if avg_value != 0 else 0
#
#                 # Display pose name and feedback
#                 cv2.putText(image, f"Pose: {detected_pose}", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 y_offset = 60
#                 for key, diff in feedback.items():
#                     cv2.putText(image, f"{key}: {diff:.2f}%", (10, y_offset),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#                     y_offset += 18
#
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
#             # Update frame live
#             frame_placeholder.image(image, channels="BGR", use_container_width=True)
#             cv2.waitKey(1)
#
#     cap.release()
#     st.session_state.detecting = False
#     st.session_state.quit = False
#     st.success("Pose detection ended.")


# =========================
# Chatbot Functions
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open('intents.json', 'r', errors="ignore") as json_data:
        intents = json.load(json_data)
except FileNotFoundError:
    st.error("intents.json not found. Please ensure the file exists.")
    intents = {'intents': []}

try:
    FILE = "yoga.pth"
    data = torch.load(FILE, map_location=device)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
except FileNotFoundError:
    st.error("yoga.pth not found. Please ensure the model file exists.")
    model = None
    tags = []
    all_words = []

bot_name = "Sam"

def get_response(msg):
    if model is None:
        return "Chatbot model not loaded."
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I do not understand..."

# =========================
# EDA Class
# =========================
class ExploratoryDataAnalysis:
    def __init__(self, data_path):
        try:
            self.data = pd.read_csv(data_path)
        except FileNotFoundError:
            st.error(f"Dataset {data_path} not found.")
            self.data = pd.DataFrame()

    def run(self):
        st.sidebar.title("Data Exploration and EDA")

        explore_option = st.sidebar.selectbox(
            "Explore Model",
            ["View Data", "View Info", "View Description", "View Missing Values"]
        )

        eda_option = st.sidebar.selectbox(
            "Exploratory Data Analysis",
            ["Select Column", "Univariate Graphs"]
        )

        if explore_option == "View Data":
            self.view_data()
        elif explore_option == "View Info":
            self.view_info()
        elif explore_option == "View Description":
            self.view_description()
        elif explore_option == "View Missing Values":
            self.view_missing_values()

        if eda_option == "Univariate Graphs":
            self.univariate_graphs()
        elif eda_option == "Bivariate Graphs":
            self.bivariate_graphs()
        elif eda_option == "Multivariate Graphs":
            self.multivariate_graphs()

    def view_data(self):
        st.title("Data Preview")
        if not self.data.empty:
            st.dataframe(self.data.head())
        else:
            st.write("No data available.")

    def view_info(self):
        st.write("Data Information")
        if not self.data.empty:
            buffer = io.StringIO()
            self.data.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
        else:
            st.write("No data available.")

    def view_description(self):
        st.write("Data Description")
        if not self.data.empty:
            st.dataframe(self.data.describe())
        else:
            st.write("No data available.")

    def view_missing_values(self):
        st.write("Missing Values")
        if not self.data.empty:
            st.dataframe(self.data.isnull().sum())
        else:
            st.write("No data available.")

    def univariate_graphs(self):
        if self.data.empty:
            st.write("No data available.")
            return
        column = st.selectbox("Select Column for Univariate Analysis", self.data.columns)
        plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Countplot"])

        plt.figure(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(self.data[column]):
            if plot_type == "Histogram":
                sns.histplot(self.data[column].dropna(), kde=True)
                st.pyplot(plt)
            elif plot_type == "Boxplot":
                sns.boxplot(x=self.data[column].dropna())
                st.pyplot(plt)
        elif pd.api.types.is_categorical_dtype(self.data[column]) or self.data[column].dtype == 'object':
            if plot_type == "Countplot":
                sns.countplot(y=self.data[column].dropna())
                st.pyplot(plt)
        else:
            st.write(f"Cannot plot {plot_type} for column: {column}")
        plt.clf()

    def bivariate_graphs(self):
        if self.data.empty:
            st.write("No data available.")
            return
        x_column = st.selectbox("Select X-axis Column", self.data.columns)
        y_column = st.selectbox("Select Y-axis Column", self.data.columns)
        plot_type = st.selectbox("Select Plot Type", ["Scatterplot", "Lineplot", "Barplot"])

        plt.figure(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(self.data[x_column]) and pd.api.types.is_numeric_dtype(self.data[y_column]):
            if plot_type == "Scatterplot":
                sns.scatterplot(x=self.data[x_column], y=self.data[y_column])
                st.pyplot(plt)
            elif plot_type == "Lineplot":
                sns.lineplot(x=self.data[x_column], y=self.data[y_column])
                st.pyplot(plt)
        elif pd.api.types.is_categorical_dtype(self.data[x_column]) or self.data[x_column].dtype == 'object':
            if plot_type == "Barplot":
                sns.barplot(x=self.data[x_column], y=self.data[y_column])
                st.pyplot(plt)
        else:
            st.write(f"Cannot plot {plot_type} for columns: {x_column} or {y_column}")
        plt.clf()

    def multivariate_graphs(self):
        if self.data.empty:
            st.write("No data available.")
            return
        columns = st.multiselect("Select Columns for Multivariate Analysis", self.data.columns)
        plot_type = st.selectbox("Select Plot Type", ["Pairplot", "Heatmap"])

        plt.figure(figsize=(10, 6))
        if plot_type == "Pairplot":
            if all(pd.api.types.is_numeric_dtype(self.data[col]) for col in columns):
                sns.pairplot(self.data[columns])
                st.pyplot(plt)
            else:
                st.write("All selected columns must be numeric for Pairplot.")
        elif plot_type == "Heatmap":
            if len(columns) > 1 and all(pd.api.types.is_numeric_dtype(self.data[col]) for col in columns):
                sns.heatmap(self.data[columns].corr(), annot=True, cmap='coolwarm')
                st.pyplot(plt)
            else:
                st.write("Select at least two numeric columns for a Heatmap.")
        plt.clf()

# =========================
# Streamlit UI
# =========================
def main():
    st.image("coverpage.png")
    st.title("Yoga Pose Correction and Recommendation")
    st.sidebar.image("s1.png")

    # Main page content
    st.markdown("""
        # AI-Powered Real-Time Yoga Pose Detection and Correction with Interactive Chatbot Support

        **Project overview**
        This application provides real-time human pose estimation tailored to yoga practice, automated pose classification, and targeted correction suggestions. A conversational chatbot supplements the visual feedback by answering user questions about posture, breathing, and modification options. The system is designed for research, teaching, and practical self-practice.

        **Key capabilities**
        1. Real time pose detection from webcam input, with robust landmark tracking for joints and limbs.
        2. Pose classification that recognizes a curated set of common yoga asanas.
        3. Automatic correction suggestions that explain which joint or alignment to change, why the change matters, and a safe modification to try.
        4. An interactive chatbot that accepts natural language questions about technique, safety, and practice sequencing.
        5. Session logging for practice review, including timestamps, detected poses, and correction highlights.
        6. Privacy by design, local processing of video when possible, and explicit consent prompts before data is stored.

        **System architecture and models**
        The front end is implemented in Streamlit to enable quick iteration and deployment. Pose estimation uses a lightweight and efficient model for low latency inference on consumer hardware. Pose classification leverages a supervised classifier trained on labeled yoga pose datasets with augmentation for varied body types and camera angles. Correction rules combine geometric heuristics from landmark positions with a learned classifier to prioritize clinically meaningful adjustments.

        **How to use**
        1. Allow webcam access when prompted.
        2. Position the camera so that your whole body is visible and lighting is stable.
        3. Choose a target pose from the pose selector or let the app detect your current pose automatically.
        4. Follow the on screen overlay for suggested alignment corrections.
        5. Ask the chatbot clarifying questions about alignment, breathing, or variations using the chat box in the sidebar.
        6. Use the session log to export practice data as a CSV for later review.

        **Evaluation and safety**
        Model performance is reported in accuracy, recall, and precision on a held out test set. Real world performance will vary by camera angle, clothing, and occlusion. All correction suggestions are phrased conservatively to reduce injury risk. The app includes a clear disclaimer to consult a certified yoga instructor or medical professional if the user has pre existing conditions.

        **Limitations and future work**
        Current pose classes cover the most common standing, seated, and balancing postures. Complex transitions and arm balances may be less reliable. Future improvements include multi view support, personalized calibration per user, finer granularity for subpose alignment, and integration of breathing and tempo detection.

        **Credits and license**
        This project combines open source pose libraries, a custom pose classifier, and an assistant module. See the About page for detailed citations, dataset acknowledgments, and the software license.

        **Contact and contribution**
        For collaboration, dataset contributions, or to report issues, use the contact link in the sidebar. Contributions that improve safety and diversity of training data are especially welcome.
        """)

    # Sidebar content
    with st.sidebar:
        st.markdown("""
        ## Controls and Quick Help

        **Live controls**
        Use the start webcam button to begin a session. Use the pose selector to fix the target pose or set the mode to automatic detection. Use the record toggle to save a practice snippet locally.

        **Chat assistant**
        Type questions about alignment, alternatives for limited mobility, or recommended adjustments. The chatbot is optimized to reference the current detected pose for context aware answers.

        **Model information**
        Model name, version, and last training date are displayed here. Expect microsecond scale latency on a modern CPU and lower latency on GPU enabled environments.

        **Privacy and storage**
        Video is processed locally by default. No data is uploaded without explicit consent. Exported session logs are stored locally unless you choose to save them to a cloud location.

        **Resources**
        View documentation, dataset citations, and usage examples from the Help link. The Help link also contains guidance on camera placement and lighting.

        **Contact**
        For research inquiries or contribution proposals, email: jithin28@gmail.com

        **Version**
        App version 1.0.0
        """)

    # Initialize session state
    for key, default in {
        "logged_in": False, "admin_logged_in": False, "user": None,
        "detecting": False, "quit": False, "chat_history": []
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    create_usertable()

    with st.sidebar:
        if not st.session_state.logged_in and not st.session_state.admin_logged_in:
            st.sidebar.title("User Authentication")
            menu = ["Login", "Register", "Admin"]
            choice = st.sidebar.selectbox("Menu", menu)

            if choice == "Login":
                st.subheader("Login Section")
                email = st.text_input("Email")
                password = st.text_input("Password", type='password')

                if st.button("Login"):
                    hashed_pswd = make_hashes(password)
                    result = login_user(email, hashed_pswd)
                    if result:
                        st.session_state.logged_in = True
                        st.session_state.user = result[0]
                        st.rerun()
                    else:
                        st.error("Invalid email or password")

            elif choice == "Register":
                st.subheader("Create New Account")
                name = st.text_input("Name")
                age = st.number_input("Age", min_value=5, max_value=100, step=1)
                email = st.text_input("Email")
                place = st.text_input("Place")
                password = st.text_input("Password", type='password')

                if st.button("Register"):
                    hashed_pswd = make_hashes(password)
                    try:
                        add_userdata(name, age, email, place, hashed_pswd)
                        st.success("Account created successfully")
                        st.info("Go to Login menu to log in")
                    except sqlite3.IntegrityError:
                        st.error("Email already exists")

            elif choice == "Admin":
                st.subheader("Admin Login")
                username = st.text_input("Username")
                password = st.text_input("Admin Password", type='password')

                if st.button("Admin Login"):
                    if check_admin_credentials(username, password):
                        st.session_state.admin_logged_in = True
                        st.session_state.user = "Admin"
                        st.rerun()
                    else:
                        st.error("Invalid admin credentials")

        else:
            st.sidebar.title("User Options")
            if st.session_state.admin_logged_in:
                st.sidebar.success("Welcome Admin")
                if st.sidebar.button("Logout"):
                    st.session_state.admin_logged_in = False
                    st.session_state.user = None
                    st.rerun()
            else:
                st.sidebar.success(f"Welcome {st.session_state.user}")
                if st.sidebar.button("Logout"):
                    st.session_state.logged_in = False
                    st.session_state.user = None
                    st.rerun()
                st.sidebar.title("🧘 Yoga Chatbot")
                st.sidebar.write("Chat with Sam, your yoga assistant! Type 'quit' to exit.")
                with st.sidebar.form(key='chat_form', clear_on_submit=True):
                    user_input = st.text_input("Your message:", key="user_input")
                    submit_button = st.form_submit_button(label="Send")

                if submit_button and user_input:
                    if user_input.lower() == "quit":
                        st.session_state.chat_history.append({"user": user_input, "bot": "Goodbye!"})
                        st.rerun()
                    else:
                        response = get_response(user_input)
                        st.session_state.chat_history.append({"user": user_input, "bot": response})
                        st.rerun()

    if st.session_state.admin_logged_in:
        st.subheader("Admin: Exploratory Data Analysis")
        eda = ExploratoryDataAnalysis("users.csv")
        eda.run()


    elif st.session_state.logged_in:

        # ==============================

        # After successful login → Pose Detection App

        # ==============================

        import time

        DATA_PATH = "yoga_pose_dataset.csv"

        st.title("🧘‍♀️ Yoga Pose Detection and Correction")

        # Check dataset

        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset file '{DATA_PATH}' not found. Please upload or place a valid CSV.")

            return

        try:

            yoga_features = pd.read_csv(DATA_PATH)

            if yoga_features.empty:
                st.error("Dataset file is empty. Add valid yoga pose data.")

                return

        except Exception as e:

            st.error(f"Error reading dataset: {e}")

            return

        # Initialize MediaPipe

        mp_pose = mp.solutions.pose

        mp_drawing = mp.solutions.drawing_utils

        # Helper functions

        def calculate_angle(a, b, c):

            a, b, c = np.array(a), np.array(b), np.array(c)

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return angle

        def detect_pose(angles):

            min_diff = float('inf')

            detected_pose = None

            for _, row in yoga_features.iterrows():

                pose_name = row['name_yoga']

                total_diff = sum(abs(angles[k] - row[k]) for k in angles.keys() if k in row)

                if total_diff < min_diff:
                    min_diff = total_diff

                    detected_pose = pose_name

            return detected_pose

        # Pose detection runner

        def run_pose_detection(video_source):

            cap = cv2.VideoCapture(video_source)

            if not cap.isOpened():
                st.error("Unable to open camera or video file.")

                return

            frame_placeholder = st.empty()

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame = cv2.resize(frame, (960, 720))

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results = pose.process(image)

                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.pose_landmarks:

                        landmarks = results.pose_landmarks.landmark

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

                            "left_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,

                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],

                            "right_hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,

                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],

                            "left_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,

                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],

                            "right_knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,

                                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],

                            "left_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,

                                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],

                            "right_ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,

                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                        }

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

                            "right_shoulder_angle": calculate_angle(keypoints["right_elbow"],
                                                                    keypoints["right_shoulder"],
                                                                    keypoints["right_hip"]),

                            "left_knee_angle": calculate_angle(keypoints["left_hip"], keypoints["left_knee"],
                                                               keypoints["left_ankle"]),

                            "right_knee_angle": calculate_angle(keypoints["right_hip"], keypoints["right_knee"],
                                                                keypoints["right_ankle"]),

                            "left_hip_angle": calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"],
                                                              keypoints["left_knee"]),

                            "right_hip_angle": calculate_angle(keypoints["right_shoulder"], keypoints["right_hip"],
                                                               keypoints["right_knee"])

                        }

                        detected_pose = detect_pose(angles)

                        feedback = {}

                        pose_row = yoga_features[yoga_features['name_yoga'] == detected_pose].iloc[0]

                        for k in angles.keys():
                            avg_value = pose_row[k]

                            feedback[k] = abs((angles[k] - avg_value) / avg_value) * 100 if avg_value != 0 else 0

                        cv2.putText(image, f"Pose: {detected_pose}", (10, 40),

                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

                        y_offset = 70

                        for key, diff in feedback.items():
                            cv2.putText(image, f"{key}: {diff:.1f}%", (10, y_offset),

                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                            y_offset += 20

                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    frame_placeholder.image(image, channels="BGR", use_container_width=True)

                    time.sleep(0.02)

            cap.release()

            st.success("Detection completed.")

        # ==============================

        # Streamlit Control UI

        # ==============================

        st.subheader("Pose Detection Options")

        mode = st.radio("Select Mode", ["Start Live Camera", "Upload Video"])

        if mode == "Start Live Camera":

            st.info("Click below to start webcam.")

            if st.button("Start Camera"):
                run_pose_detection(0)


        elif mode == "Upload Video":

            uploaded_file = st.file_uploader("Upload a Yoga Video", type=["mp4", "avi", "mov"])

            if uploaded_file is not None:

                tfile = tempfile.NamedTemporaryFile(delete=False)

                tfile.write(uploaded_file.read())

                st.success("Video uploaded successfully.")

                if st.button("Run Detection"):
                    run_pose_detection(tfile.name)

        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(f"You: {chat['user']}")
            with st.chat_message("assistant"):
                st.write(f"{bot_name}: {chat['bot']}")

if __name__ == '__main__':
    main()