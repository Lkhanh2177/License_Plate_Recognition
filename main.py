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
from init_dtb import init_db

# Add SORT library
from sort.sort import Sort

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
TEXT_SCALE = 1
TEXT_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)
TEXT_PADDING_X = 20
TEXT_PADDING_Y = 15
TEXT_GAP_FROM_BOX = 30

DB_PATH = "license_plates.db"

# SORT congfig
MAX_AGE = 10  # Maximum number of frames not seen before deleting the track
MIN_HITS = 3  # Minimum number of frames to acknowledge a track
IOU_THRESHOLD = 0.3 # IOU threshold allows to associate detection with track

# --- Camera Configuration (Change according to your actual camera name) ---
CAMERA_NAME = "Camera 1" # Current camera name

# --- Global variable to store tracking information ---

# Dictionary stores the final center position of each track ID: {id: (x, y)}
track_history = {}
# Dictionary stores the direction of movement of each track ID: {id: 'direction_string'}
track_direction = {}
# Dictionary stores the last camera position of each track ID: {id: camera_name}
track_camera_location = {}

# Load model
detect_plate_model = YOLO(os.path.join(MODEL_PATH, "best.pt")).to("cuda")

# Load paddleocr
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device = "gpu")


def save_or_update_realtime(plate_text, plate_origin, crop_img, track_id = None, direction = None, camera_location = None):
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
    """, (plate_text,)) # Thêm điều kiện camera_location nếu cần
    row = cursor.fetchone()

    if row:
        det_id, last_end_time = row
        if last_end_time:
            last_dt = datetime.datetime.strptime(last_end_time, "%Y-%m-%d %H:%M:%S")
            diff = (datetime.datetime.now() - last_dt).total_seconds()
            if diff <= 5:  # if within 5 seconds, update end_time
                cursor.execute("""
                    UPDATE detections SET end_time = ?, direction = ? WHERE id = ?
                """, (now, direction, det_id))
                conn.commit()
                conn.close()
                return

    # If new plate or not seen recently, insert new record
    cursor.execute("""
        INSERT INTO detections (plate_origin, plate_text, start_time, end_time, image_crop, direction, camera_location)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (plate_origin, plate_text, now, now, img_bytes, direction, camera_location))
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
    """Ensure the plate number is 9 characters long."""
    return plate.ljust(9, '*') if len(plate) < 8 else plate[:9]


# Detect plates in the image
def detect_plate(img):
    results = detect_plate_model.predict(img, conf=CONF_THRESH, save=False, save_txt=False, save_conf=False, iou=0.5)

    detections = [] # detections for SORT: [x1, y1, x2, y2, conf]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # Only add detection if conf > CONF_THRESH
            if conf > CONF_THRESH:
                detections.append([x1, y1, x2, y2, conf])

    # Convert detections to numpy array
    if len(detections) > 0:
        detections_np = np.array(detections)
    else:
        detections_np = np.empty((0, 5)) # Returns an empty array if there is no detection
    return detections_np


# Initialize PaddleOCR
def ocr_text(img_preprocess):
    text = ""

    result = ocr.predict(img_preprocess)

    for res in result:
        scores = res['rec_scores']
        if all(score > 0.6 for score in scores):  # Only consider results
            text += ''.join(res['rec_texts'])
    return text


def draw_plate_text(img, text, x1, y2, track_id=None, direction = None):
    if not text:
        return img

    # Color
    bg_color = (0, 255, 0)   # Nền xanh lá
    text_color = (255, 0, 0) # Chữ xanh dương

    # Display license plate text, id and direction
    display_text = f"{text} (ID: {track_id}) ({direction})" if track_id is not None else text

    (tw, th), _ = cv2.getTextSize(display_text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)

    bg_x1 = x1
    bg_y1 = y2
    bg_x2 = x1 + tw + 2 * TEXT_PADDING_X
    bg_y2 = y2 + th + 2 * TEXT_PADDING_Y

    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)

    text_x = x1 + TEXT_PADDING_X
    text_y = y2 + th + TEXT_PADDING_Y // 2
    cv2.putText(img, display_text, (text_x, text_y),
                TEXT_FONT, TEXT_SCALE, text_color, TEXT_THICKNESS, cv2.LINE_AA)

    return img

# Process Images
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Detect plates
    detections_np = detect_plate(img)

    # If there are detections, handle them.
    if detections_np.size > 0:
        for det in detections_np:
            x1, y1, x2, y2, conf = map(int, det) if det.size == 5 else (int(det[0]), int(det[1]), int(det[2]), int(det[3]), 0)
            # Crop plate
            plate_img = img[y1:y2, x1:x2]
            preprocess_plate = preprocess_input(plate_img)

            text = ocr_text(preprocess_plate)
            text = clean_plate_text(text)
            text = normalize_plate(text)

            print(f"Detected text for plate at ({x1}, {y1}, {x2}, {y2}): {text}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

            display_text = text
            (tw, th), _ = cv2.getTextSize(display_text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
            bg_x1 = x1
            bg_y1 = y2
            bg_x2 = x1 + tw + 2 * TEXT_PADDING_X
            bg_y2 = y2 + th + 2 * TEXT_PADDING_Y
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
            text_x = x1 + TEXT_PADDING_X
            text_y = y2 + th + TEXT_PADDING_Y // 2
            cv2.putText(img, display_text, (text_x, text_y),
                        TEXT_FONT, TEXT_SCALE, (255, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)

    # Display image with plates and text
    cv2.imshow("Detected Plates", resize_for_display(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def precropped_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    preprocess_plate = preprocess_input(img)

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

    # --- Initialize SORT tracker ---
    tracker = Sort(
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        iou_threshold=IOU_THRESHOLD
    )

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

        detections_np = detect_plate(frame)

        # --- Update tracker with detections ---
        # tracked_objects: [x1, y1, x2, y2, id]
        tracked_objects = tracker.update(detections_np)

        # OCR and processing for tracked objects
        for tracked_obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, tracked_obj)

            # Crop plate
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                 continue

            preprocess_plate = preprocess_input(plate_img)

            text = ocr_text(preprocess_plate)
            text_raw = clean_plate_text(text)
            text = normalize_plate(text_raw)
            if not text or text == "**.**.****": # Ignore if text is empty or invalid
                continue

            # --- Calculate movement direction ---
            current_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            prev_centroid = track_history.get(track_id)

            direction = "unknown"
            if prev_centroid is not None:
                dx = current_centroid[0] - prev_centroid[0]
                dy = current_centroid[1] - prev_centroid[1]

                # Threshold for determining significant movement
                threshold_x = 5
                threshold_y = 5
                abs_dx = abs(dx)
                abs_dy = abs(dy)

                # Determine direction based on dx and dy
                if abs_dx > threshold_x and abs_dy > threshold_y:
                    # Diagonal move
                    if dx > 0 and dy > 0:
                        direction = "diagonal_down_right"
                    elif dx > 0 and dy < 0:
                        direction = "diagonal_up_right"
                    elif dx < 0 and dy > 0:
                        direction = "diagonal_down_left"
                    elif dx < 0 and dy < 0:
                        direction = "diagonal_up_left"
                elif abs_dx > threshold_x and abs_dy <= threshold_y:
                    # Move mainly in X direction
                    if dx > 0:
                        direction = "left_to_right"
                    elif dx < 0:
                        direction = "right_to_left"
                elif abs_dy > threshold_y and abs_dx <= threshold_x:
                    # Move mainly in Y direction
                    if dy > 0:
                        direction = "top_to_bottom"
                    elif dy < 0:
                        direction = "bottom_to_top"
                else:
                    # No significant movement
                    direction = "stationary"

            else:
                # First time seeing this track
                direction = "unknown"

            #Update historical information and directions
            track_history[track_id] = current_centroid
            track_direction[track_id] = direction
            track_camera_location[track_id] = CAMERA_NAME


            # --- Save to DB ---
            # Get the last calculated direction
            final_direction = track_direction.get(track_id, "unknown")
            final_camera_location = track_camera_location.get(track_id, CAMERA_NAME)
            save_or_update_realtime(text, text_raw, preprocess_plate, track_id, final_direction, final_camera_location)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw text including license plate and track ID
            frame = draw_plate_text(frame, text, x1, y2, track_id, final_direction)

        # --- Calculate FPS ---
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # --- Draw FPS text on frame ---
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("License Plate Recognition", resize_for_display(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------Run--------------------------

# Initialize database
init_db()

# READ INPUT VIDEO
# video_file = os.path.join(TEST_VIDEO_PATH, "16.mp4")
# process_video(video_file)

# READ INPUT IMAGE
image_file = os.path.join(TEST_IMAGE_PATH, "25.png")

# ----regular image-----
process_image(image_file)

# ----precropped plate image-----
# precropped_plate(image_file)
