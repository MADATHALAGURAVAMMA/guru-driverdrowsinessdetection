from flask import Flask, render_template, Response, request, redirect, session
import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from pygame import mixer
import json
import os

app = Flask(__name__)
app.secret_key = "secretkey"

# -----------------------------
# FILE PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CNN_model.h5")
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")
USERS_FILE = os.path.join(BASE_DIR, "users1.json")

# -----------------------------
# INIT SOUND
# -----------------------------
mixer.init()
sound = mixer.Sound(ALARM_PATH)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.75
CLOSED_TIME = 3

cap = cv2.VideoCapture(0)

# -----------------------------
# USER SYSTEM
# -----------------------------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# -----------------------------
# CALCULATIONS
# -----------------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[5])
    B = distance.euclidean(mouth[2], mouth[4])
    C = distance.euclidean(mouth[0], mouth[3])
    return (A + B) / (2.0 * C)

# -----------------------------
# VIDEO STREAM
# -----------------------------
def generate_frames():
    closed_start = None
    alarm_on = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:

                left_eye = np.array([(int(face.landmark[i].x*w),
                                      int(face.landmark[i].y*h)) for i in LEFT_EYE])

                right_eye = np.array([(int(face.landmark[i].x*w),
                                       int(face.landmark[i].y*h)) for i in RIGHT_EYE])

                mouth = np.array([(int(face.landmark[i].x*w),
                                   int(face.landmark[i].y*h)) for i in MOUTH])

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
                mar = mouth_aspect_ratio(mouth)

                # Draw points
                for (x, y) in np.concatenate((left_eye, right_eye, mouth)):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Drowsy detection
                if ear < EAR_THRESHOLD:
                    if closed_start is None:
                        closed_start = time.time()

                    if time.time() - closed_start > CLOSED_TIME:
                        cv2.putText(frame, "DROWSY!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        if not alarm_on:
                            sound.play()
                            alarm_on = True
                else:
                    closed_start = None
                    alarm_on = False

                # Yawning
                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Yawning!", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        return redirect("/login")
    return render_template("dashboard.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()
        u = request.form["username"]
        p = request.form["password"]

        if u in users and users[u] == p:
            session["user"] = u
            return redirect("/dashboard")
        return "Invalid login"

    return render_template("login.html")

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users = load_users()
        u = request.form["username"]
        p = request.form["password"]

        users[u] = p
        save_users(users)
        return redirect("/login")

    return render_template("register.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect("/")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
