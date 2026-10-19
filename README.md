# AI Object Detection and Tracking System
### CodeAlpha Artificial Intelligence Internship Deliverable

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v11n%20%7C%20v8n-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracker-ByteTrack%20%2F%20SORT-green.svg)](https://arxiv.org/abs/2110.06864)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

---

## 📌 Project Overview
Real-time Computer Vision application that detects multi-class physical objects across video input or live webcam, draws bounding boxes with class labels and confidence percentages, and tracks each object with a persistent, unique tracking ID across video frames.

## 🚀 Quickstart Commands

### Windows PowerShell
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 🧪 Automated Testing
```bash
python test_system.py
```
