import cv2
import numpy as np
import os
import pyttsx3
import streamlit as st
from PIL import Image

# Initialize text-to-speech engine (optional for alerts)
engine = pyttsx3.init()

# Streamlit configuration
st.title("AI-Powered Surveillance Camera")
st.text("Press 'Start' to run the surveillance camera and 'Stop' to end it.")

# Specify the directory containing known faces
known_faces_dir = "C:/path/to/your/faces"

# Load known faces and their labels
known_faces = []
known_labels = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Ensure it's an image file
        img_path = os.path.join(known_faces_dir, filename)
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use OpenCV's face detector (Haar cascade or DNN)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Here, you can use the faces' coordinates for embedding features or identification
        if len(faces) > 0:  # Detect faces
            known_faces.append(faces)
            known_labels.append(os.path.splitext(filename)[0])  # Use filename (without extension) as label

# Initialize the video capture (use 0 for webcam)
camera = cv2.VideoCapture(0)

def alert_user(message):
    """Send an alert (optional: speech alert)."""
    st.warning(message)  # Display alert in Streamlit UI
    engine.say(message)  # Speak the message out loud
    engine.runAndWait()

def process_frame():
    """Process a single frame and return the result with face annotations."""
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to capture video")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop over the detected face coordinates
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # For recognition, you would compare the detected face's features with known_faces
        # For simplicity, we mark faces as "Known" or "Unknown" based on a match (this part is basic)
        name = "Unknown"
        for known_face, label in zip(known_faces, known_labels):
            # Here, you would use embeddings or other techniques for face recognition
            if any([np.array_equal(known_face, (x, y, w, h)) for (x, y, w, h) in faces]):
                name = label
                break

        # Display name above face
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Trigger an alert if an unknown person is detected
        if name == "Unknown":
            alert_user("Alert! Unknown person detected!")

    # Convert frame to PIL image for Streamlit
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Start and stop buttons for Streamlit
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

start = st.button("Start")
stop = st.button("Stop")

if start:
    st.session_state.camera_running = True
if stop:
    st.session_state.camera_running = False

if st.session_state.camera_running:
    stframe = st.empty()
    while st.session_state.camera_running:
        frame_image = process_frame()
        if frame_image is not None:
            stframe.image(frame_image, caption="AI Surveillance Camera", use_container_width=True)

# Release the camera when done
camera.release()
