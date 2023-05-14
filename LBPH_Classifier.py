import cv2
import os
from preprocessing import preprocess_image
import numpy as np

def train_lbph(dataset_path):
    face_cascade = cv2.CascadeClassifier('Model/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images = []
    labels = []

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = int(folder_name)
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            img = preprocess_image(img_path)
            faces = face_cascade.detectMultiScale(img)
            for (x,y,w,h) in faces:
                images.append(img[y:y+h,x:x+w])
                labels.append(label)

    recognizer.train(images, np.array(labels))
    recognizer.save('Model/lbph_model.xml')
