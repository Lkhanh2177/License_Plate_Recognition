# import libraries
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
from paddleocr import PaddleOCR
from ultralytics import YOLO
import time
import datetime

# Define paths
TEST_IMAGE_PATH = "test/images"
TEST_VIDEO_PATH = "test/videos"
MODEL_PATH = "model"

# Config
IMAGE_SIZE = (640, 640)
CONF_THRESH = 0.26
NUM_SAMPLES = 3
MAX_DIM = 1000 # Maximum dimension for resizing

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 2
TEXT_THICKNESS = 4
TEXT_COLOR = (0, 0, 255)
TEXT_PADDING_X = 20
TEXT_PADDING_Y = 15
TEXT_GAP_FROM_BOX = 30

DB_PATH = "license_plates.db"


# Load model
detect_plate_model = YOLO(os.path.join(MODEL_PATH, "best1.pt")).to("cuda")

# Load paddleocr
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device = "gpu")

# Function to save or update detection in the database
def save_or_update_realtime(plate_text, plate_origin, crop_img):
    """
    Save or update the detected license plate in the database.
    If the same plate appears again within 5 seconds, update the end_time.
    Otherwise, create a new record.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert image crop to bytes
    _, buffer = cv2.imencode(".jpg", crop_img)
    img_bytes = buffer.tobytes()

    # Check if this plate has appeared in the last 5 seconds
    cursor.execute("""
        SELECT id, end_time FROM detections
        WHERE plate_text = ?
        ORDER BY id DESC LIMIT 1
    """, (plate_text,))
    row = cursor.fetchone()

    if row:
        det_id, last_end_time = row
        if last_end_time:
            last_dt = datetime.datetime.strptime(last_end_time, "%Y-%m-%d %H:%M:%S")
            diff = (datetime.datetime.now() - last_dt).total_seconds()
            if diff <= 5:  # if within 5 seconds, update end_time
                cursor.execute("""
                    UPDATE detections SET end_time = ? WHERE id = ?
                """, (now, det_id))
                conn.commit()
                conn.close()
                return

    # If new plate or not seen recently, insert new record
    cursor.execute("""
        INSERT INTO detections (plate_origin, plate_text, start_time, end_time, image_crop)
        VALUES (?, ?, ?, ?, ?)
    """, (plate_origin, plate_text, now, now, img_bytes))
    conn.commit()
    conn.close()


# Function to calculate skew angle
def get_skew_angle(image):
    """
    calculate the skew angle of the license plate based on lines detected by HoughLinesP.
    Return the median angle to avoid noise.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=30, maxLineGap=10)
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:  # Just take reasonable angles
            angles.append(angle)

    return np.median(angles) if len(angles) > 0 else 0.0

# Preprocessing function
def preprocess_input(img):
    # --- Deskew ---
    angle = get_skew_angle(img)
    if abs(angle) > 1.5:  # Only rotate if skewed more than ~1.5°
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(
            img, rot_mat, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    h, w = img.shape[:2]
    scale = min(MAX_DIM / w, MAX_DIM / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return resized_img

def resize_for_display(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def crop_plate_strict(plate_img, top_crop_ratio=0.17):
    # crop top 17% of the plate to remove extra text
    h, w = plate_img.shape[:2]
    crop_y = int(h * top_crop_ratio)
    cropped = plate_img[crop_y:h, 0:w]
    return cropped

# Function to clean OCR text
def clean_plate_text(text):
    # Keep only alphanumeric characters, hyphens, and periods; convert to uppercase
    return re.sub(r'[^A-Z0-9\-\.]', '', text.upper())

# Function to normalize Vietnamese license plates
def normalize_plate(text):
    """
    Normalize Vietnamese license plates (8–9 characters).
    Format: 2 digits, 1–2 letters, 4–5 digits.
    """
    if not text:
        return "**.**.****"

    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())  # Keep only A–Z, 0–9

    # case 1: Matches the standard pattern
    match = re.match(r'^(\d{1,2})([A-Z]{1,2})(\d{3,5})$', cleaned)
    if match:
        num1, letters, num2 = match.groups()
        num1 = num1.ljust(2, '*')[:2]
        letters = letters[:2].ljust(1, '*')
        num2 = num2[:5].ljust(4, '*')
        return pad_plate(f"{num1}{letters}{num2}")

    # case 2: Missing front part (only letters + numbers, e.g. 'B62')
    match = re.match(r'^([A-Z]{1,2})(\d{2,5})$', cleaned)
    if match:
        letters, num2 = match.groups()
        num1 = "**"
        letters = letters[:2].ljust(1, '*')
        num2 = num2[:5].ljust(4, '*')
        return pad_plate(f"{num1}{letters}{num2}")

    # case 3: Missing middle part (e.g. '60480')
    match = re.match(r'^(\d{2})(\d{2,5})$', cleaned)
    if match:
        num1, num2 = match.groups()
        letters = "*"
        num1 = num1.zfill(2)[:2]

        num2 = num2[:5].ljust(5, '*')

        return pad_plate(f"{num1}{letters}{num2}")

    # case 4: Missing front & middle parts (e.g. '-G147-254.09' or letters before numbers)
    match = re.match(r'^[^A-Z0-9]*([A-Z]{1,2})(\d+)$', cleaned)
    if match:
        letters, num2 = match.groups()
        num1 = "**"
        letters = letters[:2].ljust(1, '*')
        num2 = num2[:6].ljust(4, '*')
        return pad_plate(f"{num1}{letters}{num2}")

    # case 5: Fallback
    return pad_plate(format_fallback(cleaned))


def format_fallback(text):
    """Basic fallback processing."""
    if not text:
        return "**.**.****"

    numbers = re.findall(r'\d+', text)
    letters = re.findall(r'[A-Z]+', text)

    num1 = numbers[0][:2].ljust(2, '*') if numbers else "**"
    letters_part = letters[0][:2] if letters else "*"
    num2 = numbers[1][:6].ljust(4, '*') if len(numbers) > 1 else "****"

    return f"{num1}{letters_part}{num2}"


def pad_plate(plate):
    """Ensure the plate number is 8 characters long."""
    return plate.ljust(8, '*') if len(plate) < 8 else plate[:9]


# Detect plates in the image
def detect_plate(img):
    results = detect_plate_model.predict(img, conf=CONF_THRESH, save=False, save_txt=False, save_conf=False, iou=0.5)

    plate_imgs = []
    bboxes = []


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop plate
            plate_img = img[y1:y2, x1:x2]
            plate_imgs.append(plate_img)
            bboxes.append((x1, y1, x2, y2))

            # # draw rectangle
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

            # # Confidence of plate detection
            # conf = float(box.conf[0])
            # label = f"Plate {conf:.2f}"
            # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #         TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS, cv2.LINE_AA)

    return plate_imgs, bboxes, img


# Initialize PaddleOCR
def ocr_text(img_preprocess):
    text = ""

    result = ocr.predict(img_preprocess)

    for res in result:
        scores = res['rec_scores']
        if all(score > 0.6 for score in scores):  # Only consider results
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
        preprocess_plate = crop_plate_strict(preprocess_plate)

        # cv2.imshow(f"Preprocessed Plate {i}", preprocess_plate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        text = ocr_text(preprocess_plate)
        text = clean_plate_text(text)
        text = normalize_plate(text)

        print(f"Detected text for plate {i}: {text}")

        img_with_box = draw_plate_text(img_with_box, text, x1, y2)
            
    # Display image with plates and text
    cv2.imshow("Detected Plates", preprocess_input(img_with_box))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def precropped_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    preprocess_plate = preprocess_input(img)
    preprocess_plate = crop_plate_strict(preprocess_plate)

    text = ocr_text(preprocess_plate)
    text_raw = clean_plate_text(text)
    text = normalize_plate(text_raw)

    print(f"Detected text for plate: {text}")

    cv2.imshow("Preprocessed Plate", preprocess_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process Videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return
    
    frame_count = 0
    prev_time = time.time()
    fps = 0.0
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        start_time = time.time()

        # Detect plates
        plates, bboxes, frame_with_box = detect_plate(frame)
        
        # OCR on detected plates
        for (plate, bbox) in zip(plates, bboxes):
            x1, y1, x2, y2 = bbox

            preprocess_plate = preprocess_input(plate)
            preprocess_plate = crop_plate_strict(preprocess_plate)

            text = ocr_text(preprocess_plate)
            text_raw = clean_plate_text(text)
            text = normalize_plate(text_raw)
            if not text:
                continue

            save_or_update_realtime(text, text_raw, preprocess_plate)

            frame_with_box = draw_plate_text(frame_with_box, text, x1, y2)

        # --- Calculate FPS ---
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # --- Draw FPS text on frame ---
        cv2.putText(frame_with_box, f"FPS: {fps:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("License Plate Recognition", resize_for_display(frame_with_box))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------Run--------------------------

# READ INPUT VIDEO
# video_file = os.path.join(TEST_VIDEO_PATH, "4.mov")
# process_video(video_file)

# READ INPUT IMAGE
image_file = os.path.join(TEST_IMAGE_PATH, "bien1.png")

# ----regular image-----
# process_image(image_file)

# ----precropped plate image-----
precropped_plate(image_file)

