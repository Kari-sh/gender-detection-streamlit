# 👤 Gender Detection Web App

A cute and interactive **real-time gender detection web application** built using **Deep Learning** and **Streamlit**.  
The app captures an image using the webcam, detects the face, and predicts gender with confidence.

---

## 🚀 Live Demo
🔗 https://YOUR-APP-LINK.streamlit.app  
*(Replace this after deployment)*

---

## ✨ Features
- 📸 Live webcam image capture
- 😊 Cute and user-friendly UI
- 🧠 Gender prediction using a trained deep learning model
- 📦 Face detection using Haar Cascade
- 📊 Confidence score display
- 🔁 Handles multiple detections by selecting the primary face

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Streamlit**
- **NumPy**
- **Pillow**

---

## 🧠 How It Works
1. User clicks **“Okay, let’s try!”**
2. Webcam opens and captures an image
3. Face is detected using Haar Cascade
4. Detected face is resized and normalized
5. A trained MobileNet-based model predicts gender
6. Result is displayed with confidence on the image

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
