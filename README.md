# License Plate Recognition
This repository provides a practical guide and implementation for **Vietnamese Vehicle License Plate Recognition** using **Yolov8** and **PaddleOCR**.

## Installation
git clone https://github.com/Lkhanh2177/License_Plate_Recognition.git
cd License_Plate_Recognition

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Lkhanh2177/License_Plate_Recognition.git
cd License_Plate_Recognition
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Install PaddleOCR
Follow the official PaddleOCR installation guide here:
👉 https://www.paddleocr.ai/latest/en/version3.x/installation.html

Note:

* If you have CUDA (GPU) installed, you can keep the code as-is for best performance.

* If you want to run on CPU, change the device argument in the PaddleOCR initialization to "cpu".

* GPU is recommended for better OCR speed and accuracy.

## Dataset
The dataset used for training and testing was obtained from Roboflow:
👉 https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk

## Run License Plate Recognition
### 1️⃣ Run the main script
```bash
python main.py
```
### 2️⃣ Choose between image or video recognition
Inside the main.py, you can comment/uncomment the relevant sections:

* Image recognition – for processing static images.
    * Regular image: detects and recognizes plates directly from full vehicle images.
    * Precropped plate image: recognizes text from already cropped license plate images.

* Video recognition – for processing real-time or recorded videos.

## Result
![alt text](https://github.com/Lkhanh2177/License_Plate_Recognition/blob/main/result/1.png)
 


