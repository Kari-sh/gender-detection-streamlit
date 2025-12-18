import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Can my model detect you? 👀",
    page_icon="🤖",
    layout="centered"
)

# ---------------- SESSION STATE ----------------
if "started" not in st.session_state:
    st.session_state.started = False

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;'>🤖 Can my model detect you?</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Let’s see if my AI can guess your gender 😄</p>",
    unsafe_allow_html=True
)
st.write("")

# ---------------- START BUTTON ----------------
if not st.session_state.started:
    if st.button("✨ Okay, let’s try!"):
        st.session_state.started = True
        st.rerun()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_gender_model():
    return load_model("gender_mobilenet224.keras")

model = load_gender_model()
classes = ["Man", "Woman"]

# ---------------- LOAD FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- CAMERA & PREDICTION ----------------
if st.session_state.started:
    st.subheader("📸 Take a photo")

    camera_image = st.camera_input("Smile 😄")

    if camera_image is not None:
        # 🌸 Cute waiting message
        with st.spinner("🧸 Wait a little baby… my AI brain is thinking 🧠✨"):
            image = Image.open(camera_image).convert("RGB")
            img = np.array(image)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if face_cascade.empty():
                st.error("❌ Face detector not loaded. Check XML file.")
            else:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(60, 60)
                )

                if len(faces) == 0:
                    st.warning("😕 No face detected. Try better lighting and face the camera.")
                else:
                    # ✅ Keep only the largest face
                    faces = sorted(
                        faces,
                        key=lambda x: x[2] * x[3],  # width * height
                        reverse=True
                    )
                    faces = [faces[0]]

                    for (x, y, w, h) in faces:
                        face = img[y:y+h, x:x+w]
                        face = cv2.resize(face, (224, 224))
                        face = face / 255.0
                        face = np.expand_dims(face, axis=0)

                        pred = model.predict(face, verbose=0)[0]
                        idx = np.argmax(pred)
                        label = classes[idx]
                        conf = pred[idx] * 100

                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(
                            img,
                            f"{label} ({conf:.1f}%)",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                    st.image(
                        img,
                        caption=f"👤 Prediction: {label} ({conf:.1f}%)",
                        width=600
                    )

                    st.success("✨ That was fun, right?")
