import face_recognition
import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # speed-up factor

    def load_encoding_images(self, images_path):
        for file_name in os.listdir(images_path):
            if file_name.endswith(('.jpg', '.png')):
                img_path = os.path.join(images_path, file_name)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    encoding = encodings[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(os.path.splitext(file_name)[0])
                else:
                    print(f"⚠️ No face found in {file_name}, skipping...")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, face_locations)
        face_names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"
            if True in matches:
                idx = matches.index(True)
                name = self.known_face_names[idx]
            face_names.append(name)

        # scale back up face locations
        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names





