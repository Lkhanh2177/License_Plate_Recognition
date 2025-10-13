# import libraries
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Define paths
TEST_IMAGE_PATH = "test/images"
TEST_VIDEO_PATH = "test/videos"
MODEL_PATH = "model"

# Config
IMAGE_SIZE = (640, 640)
CONF_THRESH = 0.25
NUM_SAMPLES = 3
MAX_DIM = 1000 # Maximum dimension for resizing

# Load model
detect_plate_model = YOLO(os.path.join(MODEL_PATH, "best1.pt")).to("cuda")

# Preprocessing function
def preprocess_input(img):
    h, w = img.shape[:2]
    scale = min(MAX_DIM / w, MAX_DIM / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return resized_img


# Detect plates in the image
def detect_plate(img):
    results = detect_plate_model.predict(img, conf=CONF_THRESH, save=False, save_txt=False, save_conf=False)

    plate_imgs = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate_img = img[y1:y2, x1:x2]
            plate_imgs.append(plate_img)

            # draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

            # If there is a label or confidence
            conf = float(box.conf[0])
            label = f"Plate {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 2, cv2.LINE_AA)

    return plate_imgs


# Initialize PaddleOCR
def ocr_text(img_preprocess):
    text = ""

    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    result = ocr.predict(img_preprocess)

    for res in result:
        text += ''.join(res['rec_texts'])

    return text


# Read input image
img = cv2.imread(f'{TEST_IMAGE_PATH}/4.png')

# Detect plates
detected_plates = detect_plate(img)

# OCR on detected plates
for i, plate in enumerate(detected_plates):
    preprocess_plate = preprocess_input(plate)

    text = ocr_text(preprocess_plate)

    print(f"Detected text for plate {i}:", text)