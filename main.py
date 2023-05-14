import cv2
import os
from LBPH_Classifier import train_lbph
from face_recognition import recognize_faces

DATASET_PATH = 'dataset'
LBPH_MODEL_PATH = 'Model/lbph_model.xml'
STUDENT_CSV_PATH = 'student.csv'

if not os.path.exists(LBPH_MODEL_PATH):
    train_lbph(DATASET_PATH)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    result = recognize_faces(frame, LBPH_MODEL_PATH, STUDENT_CSV_PATH)
    if result is None:
        print('No faces detected.')
    else:
        name, id, confidence = result
        print(f'face detected.of Name : {name} ID : {id}, Confidence : {confidence}')

    cv2.imshow('Face Recognition System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
