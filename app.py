import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from collections import Counter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Gesture Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Cards */
    .pred-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .pred-label {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
    }
    .conf-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 4px 18px;
        border-radius: 99px;
        margin-top: 6px;
    }
    .info-box {
        background: rgba(96,165,250,0.12);
        border-left: 4px solid #60a5fa;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        color: #e0e7ff;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    /* Progress bar override */
    .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #2563eb); }

    /* Metric tiles */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="metric-container"] label { color: #a5b4fc !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants matching your training setup ──────────────────────────────────────
SEQUENCE_LENGTH = 30
POSE_DIM  = 33 * 4          # 132
HAND_DIM  = 21 * 3          # 63
FEATURE_DIM = POSE_DIM + HAND_DIM * 2  # 258

CLASS_NAMES = ['Beautiful', 'expensive', 'famous', 'healthy', 'loud', 'soft', 'strong']
CLASS_EMOJIS = {
    'Beautiful': '👌', 'expensive': '💎', 'famous': '⭐',
    'healthy': '💪', 'loud': '📢', 'soft': '🌸', 'strong': '🏋️',
}

# ── Load model & labels ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    import tensorflow as tf
    model, labels = None, CLASS_NAMES
    model_path  = "model.h5"
    labels_path = "labels_2.npy"
    errors = []

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            errors.append(f"Model load error: {e}")
    else:
        errors.append(f"model.h5 not found (place it alongside app.py)")

    if os.path.exists(labels_path):
        try:
            labels = list(np.load(labels_path, allow_pickle=True))
        except Exception as e:
            errors.append(f"Labels load error: {e}")

    return model, labels, errors

@st.cache_resource(show_spinner=False)
def load_mediapipe():
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return holistic, mp_holistic, mp_drawing

# ── Keypoint extraction (matches training) ──────────────────────────────────────
def extract_keypoints(results):
    pose  = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                      for lm in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(POSE_DIM)

    lh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(HAND_DIM)

    rh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(HAND_DIM)

    return np.concatenate([pose, lh, rh])

# ── Draw landmarks on frame ─────────────────────────────────────────────────────
def draw_landmarks(frame, results, mp_holistic, mp_drawing):
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,110,10),  thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1),
        )
    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76),  thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=1),
        )
    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1),
        )
    return frame

# ── Process a video file → sequence of keypoints ────────────────────────────────
def process_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30

    keypoints_seq = []
    annotated_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        kp = extract_keypoints(results)
        keypoints_seq.append(kp)

        # Draw landmarks
        annotated = draw_landmarks(frame.copy(), results, mp_holistic_global, mp_drawing_global)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated_frames.append(annotated_rgb)

    cap.release()

    # Pad / trim to SEQUENCE_LENGTH
    seq = np.array(keypoints_seq)
    if len(seq) == 0:
        return None, total_frames

    if len(seq) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(seq), FEATURE_DIM))
        seq = np.vstack([seq, pad])
    else:
        # sample evenly
        indices = np.linspace(0, len(seq) - 1, SEQUENCE_LENGTH, dtype=int)
        seq = seq[indices]
        annotated_frames = [annotated_frames[i] for i in indices]

    return seq, total_frames, annotated_frames, fps

# ── Predict ─────────────────────────────────────────────────────────────────────
def predict(model, seq, labels):
    inp = seq[np.newaxis, ...]           # (1, 30, 258)
    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = labels[idx] if idx < len(labels) else f"Class {idx}"
    conf  = float(probs[idx])
    return label, conf, probs

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 ISL Recognizer")
    st.markdown("**Indian Sign Language** gesture classification using MediaPipe + LSTM")
    st.divider()

    st.markdown("### ℹ️ Supported Gestures")
    for c in CLASS_NAMES:
        st.markdown(f"{CLASS_EMOJIS.get(c,'🤚')} **{c}**")

    st.divider()
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    show_landmarks = st.toggle("Show MediaPipe landmarks", value=True)
    show_all_probs = st.toggle("Show all class probabilities", value=True)
    st.divider()
    st.caption("Place `model.h5` and `labels_2.npy` in the same folder as app.py")

# ── Load resources ──────────────────────────────────────────────────────────────
with st.spinner("Loading model & MediaPipe…"):
    model, labels, load_errors = load_model_and_labels()
    holistic, mp_holistic_global, mp_drawing_global = load_mediapipe()

# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown("# 🤟 ISL Gesture Recognition")
st.markdown("Upload a sign-language video and get an instant prediction.")

# Show load warnings
if load_errors:
    for err in load_errors:
        st.warning(f"⚠️ {err}")
    if model is None:
        st.info(
            "**Demo mode**: Model not found. Upload a video to see landmark extraction; "
            "predictions require `model.h5` in the same directory."
        )

st.divider()

# ── Upload ───────────────────────────────────────────────────────────────────────
col_up, col_info = st.columns([2, 1])

with col_up:
    uploaded = st.file_uploader(
        "Upload a gesture video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Short video (1–5 seconds) of an ISL sign works best.",
    )

with col_info:
    st.markdown('<div class="info-box">'
                '📌 <b>Tips for best results</b><br>'
                '• 1–5 second clips<br>'
                '• Good lighting<br>'
                '• Hands clearly visible<br>'
                '• Neutral background'
                '</div>', unsafe_allow_html=True)

# ── Main processing ──────────────────────────────────────────────────────────────
if uploaded is not None:
    # Save to temp file
    suffix = os.path.splitext(uploaded.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Show original video
    st.markdown("### 🎬 Uploaded Video")
    st.video(tmp_path)

    st.markdown("### 🔍 Processing…")
    progress_bar = st.progress(0, text="Extracting keypoints…")

    try:
        result = process_video(tmp_path, holistic)

        if result is None or result[0] is None:
            st.error("Could not read the video. Please try a different file.")
        else:
            seq, total_frames, annotated_frames, fps = result
            progress_bar.progress(70, text="Running prediction…")

            # ── Prediction ────────────────────────────────────────────────────
            if model is not None:
                label, conf, probs = predict(model, seq, labels)
                progress_bar.progress(100, text="Done!")
                time.sleep(0.3)
                progress_bar.empty()

                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")

                # Main result card
                emoji = CLASS_EMOJIS.get(label, '🤚')
                st.markdown(
                    f'<div class="pred-card">'
                    f'<div style="font-size:3.5rem">{emoji}</div>'
                    f'<div class="pred-label">{label}</div><br>'
                    f'<span class="conf-badge">Confidence: {conf*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if conf < conf_threshold:
                    st.warning(
                        f"⚠️ Confidence ({conf*100:.1f}%) is below your threshold "
                        f"({conf_threshold*100:.0f}%). The gesture may be unclear."
                    )

                # Metrics row
                st.markdown("")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted Gesture", label)
                m2.metric("Confidence", f"{conf*100:.1f}%")
                m3.metric("Frames Processed", total_frames)
                m4.metric("Sequence Length", SEQUENCE_LENGTH)

                # All class probabilities
                if show_all_probs:
                    st.markdown("#### 📊 All Class Probabilities")
                    sorted_idx = np.argsort(probs)[::-1]
                    for i in sorted_idx:
                        lbl = labels[i] if i < len(labels) else f"Class {i}"
                        p   = float(probs[i])
                        e   = CLASS_EMOJIS.get(lbl, '🤚')
                        col_lbl, col_bar = st.columns([1, 3])
                        with col_lbl:
                            st.markdown(f"{e} **{lbl}**")
                        with col_bar:
                            st.progress(p, text=f"{p*100:.1f}%")

            else:
                # Demo mode – no model
                progress_bar.progress(100, text="Landmark extraction complete.")
                progress_bar.empty()
                st.info("📌 Prediction skipped — `model.h5` not loaded. Showing landmark frames below.")

            # ── Annotated frames preview ────────────────────────────────────
            if show_landmarks and annotated_frames:
                st.markdown("---")
                st.markdown("### 🦴 MediaPipe Landmark Preview")
                st.caption("Sampled frames with pose & hand landmarks overlaid")

                n_preview = min(8, len(annotated_frames))
                step = max(1, len(annotated_frames) // n_preview)
                preview_frames = annotated_frames[::step][:n_preview]

                cols = st.columns(min(4, len(preview_frames)))
                for i, frame in enumerate(preview_frames):
                    with cols[i % 4]:
                        st.image(frame, caption=f"Frame {i*step+1}", use_container_width=True)

    except Exception as e:
        progress_bar.empty()
        st.error(f"Error during processing: {e}")
        st.exception(e)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

else:
    # ── Empty state ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #a5b4fc;">
        <div style="font-size: 4rem;">🤟</div>
        <h3 style="color:#c4b5fd;">Upload a video to get started</h3>
        <p>Supported formats: MP4, AVI, MOV, MKV, WebM</p>
    </div>
    """, unsafe_allow_html=True)

    # Show class grid as teaser
    st.markdown("### 🗂️ Recognizable Gestures")
    cols = st.columns(4)
    for i, c in enumerate(CLASS_NAMES):
        with cols[i % 4]:
            st.markdown(
                f'<div class="pred-card" style="padding:1rem;">'
                f'<div style="font-size:2rem">{CLASS_EMOJIS.get(c,"🤚")}</div>'
                f'<div style="color:#e0e7ff;font-weight:600;margin-top:4px">{c}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )