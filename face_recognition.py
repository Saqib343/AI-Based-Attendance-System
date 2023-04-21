import cv2
import numpy as np
import csv

def recognize_faces(frame, model_path, csv_path):
    face_cascade = cv2.CascadeClassifier('Model/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        students = {row['id']: row['name'] for row in reader}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return [], [], []

    names = []
    labels = []
    confidences = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(roi_gray)
        if confidence < 60:
            name = students[str(label)] if str(label) in students else "Unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{name} ({label})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    if row['id'] == str(label):
                        row['status'] = 'P'
                    rows.append(row)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return names, labels, confidences

