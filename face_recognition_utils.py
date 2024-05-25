import cv2
import dlib
import numpy as np
import os
import face_recognition



FACE_DIR = "./faces"
if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

def load_known_faces():
    known_faces = {}
    for filename in os.listdir(FACE_DIR):
        if filename.endswith('_data.npy'):
            encodings = np.load(os.path.join(FACE_DIR, filename), allow_pickle=True)
            known_faces[filename.split('_')[0]] = encodings
    return known_faces



def capture_face_data(name):
    print(f"[INFO] Starting face data capture for {name}...")
    cap = cv2.VideoCapture(0)
    all_encodings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame!")
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            all_encodings.append(face_encoding)
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow(f"Face Data Collection for {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not all_encodings:
        print("[ERROR] No face data captured. Trying again...")
        capture_face_data(name)
        return

    data_path = os.path.join(FACE_DIR, f"{name}_data.npy")
    np.save(data_path, all_encodings)
    print(f"[INFO] Face data for {name} captured and saved as {data_path}!")


def recognize_face_video():
    print("[INFO] Starting face recognition in video stream...")

    known_faces = {}
    for filename in os.listdir(FACE_DIR):
        if filename.endswith('_data.npy'):
            encodings = np.load(os.path.join(FACE_DIR, filename), allow_pickle=True)
            known_faces[filename.split('_')[0]] = encodings

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame!")
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

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_face_in_image_stream(image_array):
    # Load known face encodings
    known_faces = {}
    for filename in os.listdir(FACE_DIR):
        if filename.endswith('_data.npy'):
            encodings = np.load(os.path.join(FACE_DIR, filename), allow_pickle=True)
            known_faces[filename.split('_')[0]] = encodings

    # Decode the image array to OpenCV format
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Detect faces
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Annotate faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        label = "Unknown"
        for name, known_face_encodings in known_faces.items():
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                label = name
                break

        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)

    # Convert back to RGB for display in Streamlit
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image


def recognize_face_in_image(image_path):
    print(f"[INFO] Starting face recognition in image {image_path}...")

    known_faces = {}
    for filename in os.listdir(FACE_DIR):
        if filename.endswith('_data.npy'):
            encodings = np.load(os.path.join(FACE_DIR, filename), allow_pickle=True)
            known_faces[filename.split('_')[0]] = encodings

    image = cv2.imread(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        label = "Unknown"
        for name, known_face_encodings in known_faces.items():
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                label = name
                break

        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 13, (255, 0, 0), 3)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Recognition in Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def capture_face_data_from_folder(dir_path):
    for subdir, _, files in os.walk(dir_path):
        person_name = os.path.basename(subdir)
        all_encodings = []

        for file in files:
            image_path = os.path.join(subdir, file)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if not face_encodings:
                    print(f"[WARNING] No faces found in {image_path}. Skipping...")
                    continue

                all_encodings.extend(face_encodings)
            except Exception as e:
                print(f"[ERROR] Error processing image {image_path}. Error: {e}")

        if all_encodings:
            data_path = os.path.join(FACE_DIR, f"{person_name}_data.npy")
            np.save(data_path, all_encodings)
            print(f"[INFO] Face data for {person_name} captured and saved as {data_path}!")

def check_face_in_video(face_data_path, video_path):
    print(f"[INFO] Loading face data from {face_data_path}")
    known_face_encodings = np.load(face_data_path, allow_pickle=True)

    print(f"[INFO] Opening video file {video_path}")
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        frame_number += 1
        if not ret:
            print(f"[INFO] Reached end of video or failed to read frame {frame_number}. Exiting...")
            break

        print(f"[INFO] Processing frame {frame_number}")
        frame_face_encodings = face_recognition.face_encodings(frame)

        for encoding in frame_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            if True in matches:
                print(f"[INFO] Face found in frame {frame_number}!")
                cap.release()
                cv2.destroyAllWindows()
                return True

    print("[INFO] Face not found in video.")
    cap.release()
    cv2.destroyAllWindows()
    return False



