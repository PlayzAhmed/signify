# Signify
**Signify** is a real-time AI-powered sign language translator that converts sign gestures into text and spoken language. It supports both single and double-hand models, and includes support for ASL and ArSL.

## 🌟 Features

- Real-time webcam-based hand tracking
- Supports American and Arabic Sign Language (ASL, ArSL)
- Works with both single-hand and double-hand gestures
- Smart mapping of gestures to full phrases
- Text-to-text and text-to-speech translation using Google Translate
- GUI built with `CustomTkinter`

## 🧠 Tech Stack

- **Python**
- TensorFlow / Keras
- OpenCV
- MediaPipe
- Google Translate API (`googletrans`)
- CustomTkinter

## 📁 Project Structure

```
signify/
│
├── app.py # Main application GUI and logic
├── models/ # Pre-trained models (.h5) and encoders (.pkl)
│ ├── asl_single_hand_model.keras
│ ├── asl_double_hand_model.keras
│ ├── asl_single_hand_label_encoder.pkl
│ └── asl_double_hand_label_encoder.pkl
├── requirements.txt # List of dependencies
└── README.md
```


## ⚙️ Installation

```bash
git clone https://github.com/PlayzAhmed/signify.git
cd signify
pip install -r requirements.txt
python app.py
```

## 📦 Requirements

You can install all required packages with:
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install tensorflow mediapipe opencv-python customtkinter pillow numpy scikit-learn googletrans==4.0.0rc1
```

## 🎓 Author
**Ahmed Ismail**

Passionate high school student & AI enthusiast.

- [LinkedIn](https://www.linkedin.com/in/ahmed-mohammed-853b53300/)
