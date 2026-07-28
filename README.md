# 🛡️ SmartShield AI

SmartShield AI is a document privacy protection system developed using **Python, Streamlit, OpenCV, and YOLO**. The project automatically detects sensitive information such as **faces** and **handwritten signatures** and redacts them using Gaussian blur before the document is shared.

This project was built to learn how computer vision and deep learning can be applied to solve real-world privacy and document security problems.

---

## Features

- Upload document images
- Optional perspective correction
- Face detection using OpenCV
- Signature detection using a YOLO model
- Adjustable blur intensity
- Automatic privacy redaction
- Compliance summary
- Download redacted image
- Download audit log

---

## Project Pipeline

```
Upload Document
       │
       ▼
Convert Image to OpenCV Format
       │
       ▼
(Optional) Perspective Correction
       │
       ▼
Face Detection
       │
       ▼
Signature Detection (YOLO)
       │
       ▼
Apply Gaussian Blur to Sensitive Regions
       │
       ▼
Generate Audit Log
       │
       ▼
Display Results
       │
       ▼
Download Redacted Image & Audit Log
```

---

## Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- Ultralytics YOLO
- PyTorch

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
SMARTSHIELD-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── backend/
├── models/
└── venv/
```

---

## Learning Outcome

Through this project, I gained practical experience in:

- Image processing using OpenCV
- Object detection using YOLO
- Building an interactive web application with Streamlit
- Integrating multiple computer vision modules into a single workflow
- Designing a simple privacy-preserving document processing pipeline

---

