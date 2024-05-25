# streamlit_app.py
import face_recognition
import streamlit as st
import cv2
import numpy as np
import os

from PIL import Image

from face_recognition_utils import *

FACE_DIR = "./faces"
if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

def recognize_face_video():
    st.title("Real-time Face Recognition")
    known_faces = load_known_faces()
    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            label = "Unknown"
            for name, known_face_encodings in known_faces.items():
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True in matches:
                    label = name
                    break

            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame_window.image(frame)



    cap.release()




st.title('Helos.ai Face Model App')
option = st.sidebar.selectbox(
    'Choose an option:',
    ('Capture Face Data', 'Recognize Face in Real-time Video', 'Recognize Face in Image', 'Capture Face Data from Folder', 'Check Face in Video')
)

if option == 'Capture Face Data':
    st.header('Capture Face Data')
    name = st.text_input("Enter the name of the person:")
    if st.button('Start Capturing'):
        # Call your capture_face_data function
        capture_face_data(name)

elif option == 'Recognize Face in Real-time Video':
    st.header('Recognize and Tag Faces in Real-time Video')
    if st.button('Start Video Recognition'):
        recognize_face_video()

elif option == 'Recognize Face in Image':

        st.header('Helos.ai Recognize Face in an Image')
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
        if image_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            if st.button('Recognize Faces'):
                # Process the image and recognize faces
                annotated_image = recognize_face_in_image_stream(file_bytes)
                # Display the annotated image
                st.image(annotated_image, channels="RGB", use_column_width=True)


elif option == 'Capture Face Data from Folder':
    st.header('Capture Face Data from a Folder')
    dir_path = st.text_input("Enter the directory path:")
    if st.button('Capture Data'):
        capture_face_data_from_folder(dir_path)

elif option == 'Check Face in Video':
    st.header('Check if a Face Exists in a Video')
    face_image_path = st.text_input("Enter the path to the face data file:")
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    if video_file is not None and face_image_path:
        if st.button('Check Video'):
            result = check_face_in_video(face_image_path, video_file)
            if result:
                st.success("Face exists in video")
            else:
                st.error("Face not found in video")

# More app logic...
