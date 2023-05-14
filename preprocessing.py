import os
import cv2
import numpy as np

# Define the path to the dataset folder
dataset_path = "dataset/"

# Loop over each subdirectory in the dataset folder
for subdir in os.listdir(dataset_path):
    subdir_path = os.path.join(dataset_path, subdir)

    # Loop over each image file in the subdirectory
    for filename in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, filename)

        # Check if file is an image file
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):

            # Read the image file
            image = cv2.imread(file_path)

            # Remove the background using GrabCut algorithm
            mask = np.zeros(image.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            (x, y, w, h) = cv2.boundingRect(cv2.findNonZero(cv2.Canny(image, 100, 200)))
            cv2.grabCut(image, mask, (x, y, w, h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            image = image * mask2[:, :, np.newaxis]

            # Remove noise using bilateral filter
            image = cv2.bilateralFilter(image, 9, 75, 75)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Adjust brightness
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Increase sharpness
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            image = cv2.filter2D(image, -1, sharpen_kernel)

            # Save the processed image
            cv2.imwrite(file_path, gray)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray