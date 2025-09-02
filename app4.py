import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import time

st.title("🪄 Smooth Body Vanish Effect (Webcam)")
st.write("Cover your face with any object → your body will fade smoothly into the background!")

# Initialize session state
if "cap" not in st.session_state:
    st.session_state.cap = None
if "background" not in st.session_state:
    st.session_state.background = None
if "running" not in st.session_state:
    st.session_state.running = False
if "alpha" not in st.session_state:
    st.session_state.alpha = 0.0  # blending factor (0 = visible, 1 = invisible)

# Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Mediapipe segmentation
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶ Start Webcam"):
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.running = True
with col2:
    if st.button("📷 Capture Background"):
        if st.session_state.cap and st.session_state.cap.isOpened():
            ret, frame = st.session_state.cap.read()
            if ret:
                st.session_state.background = cv2.flip(frame, 1)
                st.success("✅ Background captured!")
with col3:
    if st.button("🛑 Stop Webcam"):
        st.session_state.running = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.alpha = 0.0
        st.success("🛑 Webcam stopped.")

# Display area
frame_window = st.empty()

# Auto-refresh loop
while st.session_state.running and st.session_state.cap and st.session_state.cap.isOpened():
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("⚠ Failed to access webcam. Try restarting.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if st.session_state.background is not None:
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Decide vanish state
        vanish = False
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_region = frame[y:y+h, x:x+w]
            if np.var(face_region) < 3000:  # face is covered
                vanish = True
        else:
            vanish = True  # no face → vanish

        # Smooth alpha transition
        if vanish:
            st.session_state.alpha = min(1.0, st.session_state.alpha + 0.05)  # fade out
        else:
            st.session_state.alpha = max(0.0, st.session_state.alpha - 0.05)  # fade in

        # Get segmentation mask
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmentor.process(rgb_frame)
        mask = results.segmentation_mask > 0.5

        # Composite with background
        body = np.where(mask[..., None], frame, st.session_state.background)
        blended = cv2.addWeighted(frame, 1 - st.session_state.alpha, body, st.session_state.alpha, 0)

    else:
        blended = frame

    # Update frame
    frame_window.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    time.sleep(0.03)
