# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import os
from face_recognition_utils import capture_face_data, recognize_face_video, recognize_face_in_image, \
    capture_face_data_from_folder, check_face_in_video, recognize_face_in_image_stream

FACE_DIR = "./faces"
if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

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
