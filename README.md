# 🤟 ISL Gesture to Text Converter

A real-time Artificial Intelligence application that translates **Indian Sign Language (ISL) gestures into text** using **Computer Vision** and **Deep Learning**.

This project aims to enhance accessibility and bridge communication gaps for individuals with hearing or speech impairments through automated gesture recognition.

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit">
</p>

---

## 📌 Overview

The system captures hand gestures through a webcam, processes them using image preprocessing techniques, and predicts the corresponding ISL character using a trained Convolutional Neural Network (CNN) model.

The predicted gesture is then converted into readable text in real time.

---

## ✨ Features

✔️ Real-time ISL gesture recognition
✔️ Webcam-based live detection
✔️ Gesture-to-text conversion
✔️ Deep Learning powered classification
✔️ Interactive Streamlit interface
✔️ Efficient image preprocessing pipeline

---

## 🛠️ Tech Stack

| Technology         | Usage                  |
| ------------------ | ---------------------- |
| Python             | Core Programming       |
| OpenCV             | Image Processing       |
| TensorFlow / Keras | Deep Learning          |
| NumPy              | Numerical Operations   |
| Streamlit          | Frontend Interface     |
| CNN                | Gesture Classification |

---

## 📂 Repository Structure

```bash
isl-gesture-to-text/
│
├── .streamlit/            # Streamlit configuration
├── app.py                 # Main application
├── check_labels.py        # Label verification
├── check_model.py         # Model inspection
├── inspect_data.py        # Dataset inspection
├── labels_2.npy           # Saved labels
├── model.h5               # Trained model
├── requirements.txt       # Dependencies
├── runtime.txt            # Runtime configuration
├── test_tf.py             # TensorFlow testing
├── verify.py              # Verification script
└── .gitignore
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Shipra-Jaiswal/isl-gesture-to-text.git
```

### Navigate to Project Directory

```bash
cd isl-gesture-to-text
```

### Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧠 Working Pipeline

```text
Webcam Input
     ↓
Image Preprocessing
     ↓
Feature Extraction
     ↓
CNN Model Prediction
     ↓
ISL Gesture Recognition
     ↓
Text Output Generation
```

---

## 📸 Demo

Add project screenshots or GIF demonstrations here.

Suggested screenshots:

* Live webcam feed
* Gesture prediction result
* Streamlit interface
* Text generation output

---

## 🎯 Future Scope

* Continuous gesture recognition
* Sentence generation support
* Speech synthesis integration
* Mobile/Web deployment
* Multi-language translation

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome.

To contribute:

```bash
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to GitHub
5. Create a Pull Request
```
🔗 GitHub: https://github.com/Shipra-Jaiswal

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
