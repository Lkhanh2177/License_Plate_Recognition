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

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 2
TEXT_THICKNESS = 4
TEXT_COLOR = (0, 0, 255)
TEXT_PADDING_X = 20
TEXT_PADDING_Y = 15
TEXT_GAP_FROM_BOX = 30


# Load model
detect_plate_model = YOLO(os.path.join(MODEL_PATH, "best1.pt")).to("cuda")

# Load paddleocr
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device = "gpu")


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
    bboxes = []


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop plate
            plate_img = img[y1:y2, x1:x2]
            plate_imgs.append(plate_img)
            bboxes.append((x1, y1, x2, y2))

            # draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

            # Confidence of plate detection
            conf = float(box.conf[0])
            label = f"Plate {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS, cv2.LINE_AA)

    return plate_imgs, bboxes, img


# Initialize PaddleOCR
def ocr_text(img_preprocess):
    text = ""

    result = ocr.predict(img_preprocess)

    for res in result:
        text += ''.join(res['rec_texts'])

    return text


def draw_plate_text(img, text, x1, y2):
    if not text:
        return img
    
    (tw, th), _ = cv2.getTextSize(text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)

    # Padding
    bg_x1 = x1 - TEXT_PADDING_X
    bg_y1 = y2 + TEXT_GAP_FROM_BOX - TEXT_PADDING_Y
    bg_x2 = x1 + tw + TEXT_PADDING_X
    bg_y2 = y2 + TEXT_GAP_FROM_BOX + th + TEXT_PADDING_Y

    # Minimize going out of image bounds
    bg_x1 = max(0, bg_x1)
    bg_y2 = min(img.shape[0], bg_y2)

    # Draw white background rectangle
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)

    # Put text over the white rectangle
    text_y = y2 + TEXT_GAP_FROM_BOX + th
    cv2.putText(img, text, (x1, text_y),
                TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    
    return img


# Process Images
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Detect plates
    plates, bbox, img_with_box = detect_plate(img)

    # OCR on detected plates
    for i, (plate, bbox) in enumerate(zip(plates, bbox)):
        x1, y1, x2, y2 = bbox
        preprocess_plate = preprocess_input(plate)
        text = ocr_text(preprocess_plate)
        print(f"Detected text for plate {i}: {text}")

        img_with_box = draw_plate_text(img_with_box, text, x1, y2)
            
    # Display image with plates and text
    cv2.imshow("Detected Plates", preprocess_input(img_with_box))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process Videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return
    

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        plates, bboxes, frame_with_box = detect_plate(frame)

        for (plate, bbox) in zip(plates, bboxes):
            x1, y1, x2, y2 = bbox
            preprocess_plate = preprocess_input(plate)
            text = ocr_text(preprocess_plate)
            frame_with_box = draw_plate_text(frame_with_box, text, x1, y2)

        cv2.imshow("License Plate Recognition", preprocess_input(frame_with_box))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------Run--------------------------

# Read input video
video_file = os.path.join(TEST_VIDEO_PATH, "2.mov")
process_video(video_file)

# Read input image
# image_file = os.path.join(TEST_IMAGE_PATH, "6.png")
# process_image(image_file)

