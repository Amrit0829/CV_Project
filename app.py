import cv2
import numpy as np
import os
import pyttsx3  # For voice alerts (optional)
import face_recognition  # For face recognition
import streamlit as st
from PIL import Image
import tempfile

# Initialize text-to-speech engine (optional for alerts)
engine = pyttsx3.init()

# Streamlit configuration
st.title("AI-Powered Surveillance Camera")
st.text("Upload a video or use webcam for surveillance.")

# Specify the directory containing known faces (update the path to the correct location)
known_faces_dir = "C:/Users/AMRIT SHYAM KADBHANE/Downloads/AI-Powered_Surveillance_Camera/AI-Powered_Surveillance_Camera/"

# Load known faces and their labels
known_faces = []
known_labels = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Ensure it's an image file
        img_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:  # Check if encoding was found
            known_faces.append(encoding[0])
            known_labels.append(os.path.splitext(filename)[0])  # Use filename (without extension) as label

# Function to process frames from the video
def process_frame(frame):
    """Process a single frame and return the result with face annotations."""
    # Convert the frame from BGR to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over the detected face encodings
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # If a match was found, find the corresponding label
        if True in matches:
            first_match_index = matches.index(True)
            name = known_labels[first_match_index]

        # Draw a rectangle around the detected face
        (top, right, bottom, left) = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Trigger an alert if an unknown person is detected
        if name == "Unknown":
            alert_user("Alert! Unknown person detected!")

    # Convert frame to PIL image for Streamlit
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Function for alert (optional: voice alert)
def alert_user(message):
    """Send an alert (optional: speech alert)."""
    st.warning(message)  # Display alert in Streamlit UI
    engine.say(message)  # Speak the message out loud
    engine.runAndWait()

# Upload video file for analysis
uploaded_video = st.file_uploader("Upload a video for surveillance", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    st.video(video_path)  # Display the video in the Streamlit app

    # Open the video for processing
    cap = cv2.VideoCapture(video_path)

    # Streamlit's real-time video processing (video playback and face recognition)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        # Process the current frame
        frame_image = process_frame(frame)
        if frame_image is not None:
            stframe.image(frame_image, caption="AI Surveillance Camera", use_container_width=True)

    # Release the video capture once done
    cap.release()
else:
    st.text("Upload a video to begin surveillance.")
