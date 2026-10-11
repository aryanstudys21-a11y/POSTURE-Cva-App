import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import pandas as pd
from PIL import Image, ImageOps
from fpdf import FPDF
import tempfile
import datetime

st.set_page_config(page_title="KINEMA-AI | Clinical", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border-left: 5px solid #3B82F6 !important;
    }
    div[data-testid="stMetricLabel"] > div { color: #94A3B8 !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"] > div { color: #F8FAFC !important; font-weight: 800 !important; }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "patient_db" not in st.session_state: st.session_state.patient_db = []

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)

def calculate_horizontal_angle(pt1, pt2):
    dx = pt1[0] - pt2[0]
    dy = pt2[1] - pt1[1] 
    return math.degrees(math.atan2(dy, abs(dx)))

def auto_orient_image(image_np):
    """Multi-pass gravity check: rotates image until the nose is detected above the shoulders."""
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    for rot in rotations:
        test_img = cv2.rotate(image_np, rot) if rot is not None else image_np
        res = pose.process(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            nose_y = lm[mp_pose.PoseLandmark.NOSE.value].y
            l_sh_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            r_sh_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            if nose_y < l_sh_y and nose_y < r_sh_y:  # Y-axis increases downward
                return test_img, res.pose_landmarks.landmark
    return image_np, None

def analyze_frame(image_np, strictness):
    h, w, _ = image_np.shape
    annotated = image_np.copy()
    
    oriented_img, lm = auto_orient_image(image_np)
    if not lm:
        return None, "Subject not detected. Ensure head and shoulders are visible."
        
    annotated = oriented_img.copy()
    h, w, _ = annotated.shape
    
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
    r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
    
    # Strict Plane Routing: Shoulders must be horizontally wide apart AND both ears highly visible
    is_coronal = abs(l_sh.x - r_sh.x) > 0.15 and (l_ear.visibility > 0.6 and r_ear.visibility > 0.6)
    
    if is_coronal:
        l_ear_px, r_ear_px = (int(l_ear.x * w), int(l_ear.y * h)), (int(r_ear.x * w), int(r_ear.y * h))
        l_sh_px, r_sh_px = (int(l_sh.x * w), int(l_sh.y * h)), (int(r_sh.x * w), int(r_sh.y * h))
        
        head_tilt = round(abs(calculate_horizontal_angle(l_ear_px, r_ear_px)), 2)
        cv2.line(annotated, l_ear_px, r_ear_px, (255, 150, 0), 4, cv2.LINE_AA)
        cv2.line(annotated, l_sh_px, r_sh_px, (0, 200, 255), 4, cv2.LINE_AA)
        
        status = "Clinically Symmetrical" if head_tilt < strictness else "Asymmetry Detected"
        return annotated, {"Plane": "Coronal (Frontal)", "Primary_Metric": "Head Tilt", "Primary_Value": head_tilt, "Status": status}
    
    else:
        side = "Left Lateral" if l_ear.visibility > r_ear.visibility else "Right Lateral"
        ear_px = (int(l_ear.x * w), int(l_ear.y * h)) if "Left" in side else (int(r_ear.x * w), int(r_ear.y * h))
        sh_px = (int(l_sh.x * w), int(l_sh.y * h)) if "Left" in side else (int(r_sh.x * w), int(r_sh.y * h))
        
        scale = abs(sh_px[1] - ear_px[1])
        offset_x = 0.08 if "Left" in side else -0.08
        c7_px = (int(sh_px[0] + (scale * offset_x)), int(sh_px[1] - (scale * 0.18)))
        
        cva = round(calculate_horizontal_angle(ear_px, c7_px), 2)
        color = (186, 82, 15) if cva >= 50 else (50, 50, 220)
        
        cv2.line(annotated, c7_px, ear_px, color, 4, cv2.LINE_AA)
        cv2.line(annotated, c7_px, (c7_px[0] + (150 if "Right" in side else -150), c7_px[1]), (200, 200, 200), 2, cv2.LINE_AA)
        
        status = "Optimal Alignment" if cva >= 50 else "Severe FHP" if cva < 40 else "FHP"
        return annotated, {"Plane": f"Sagittal ({side})", "Primary_Metric": "CVA", "Primary_Value": cva, "Status": status}

st.sidebar.markdown("### 🧬 KINEMA-AI")
page = st.sidebar.radio("Navigation", ["📸 Active Scanner", "⚙️ Settings"])

if page == "📸 Active Scanner":
    img_file = st.file_uploader("Upload Scan", type=["png", "jpg", "jpeg"])
    if img_file:
        img_np = np.array(ImageOps.exif_transpose(Image.open(img_file)))
        strictness = st.session_state.get("strictness", 4.0)
        
        with st.spinner("Executing gravity auto-orientation and plane detection..."):
            res = analyze_frame(img_np, strictness)
            
        if res[0] is not None:
            col1, col2 = st.columns([1.5, 1])
            col1.image(res[0], channels="RGB", use_container_width=True)
            col2.metric("Detected Plane", res[1]["Plane"])
            col2.metric(res[1]["Primary_Metric"], f"{res[1]['Primary_Value']}°")
            col2.metric("Diagnostic Status", res[1]["Status"])
        else:
            st.error(res[1])
elif page == "⚙️ Settings":
    st.session_state.strictness = st.slider("Coronal Tolerance (°)", 1.0, 10.0, 4.0, 0.5)
