import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
from PIL import Image

st.set_page_config(page_title="Clinical AI Vision", page_icon="🔬", layout="centered")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.65
)

def get_plane_type(landmarks):
    """Dynamically determines if the subject is facing front (Coronal) or side (Lateral)."""
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # Calculate 2D horizontal normalized width between shoulders
    shoulder_width = abs(l_sh.x - r_sh.x)
    
    # If shoulders are wide apart on the X-axis, it's a frontal view
    if shoulder_width > 0.12:
        return "Coronal"
    return "Lateral"

def calculate_angle(pt1, pt2):
    """Calculates angle relative to horizontal baseline."""
    dx = pt1[0] - pt2[0]
    dy = pt2[1] - pt1[1] 
    return math.degrees(math.atan2(dy, abs(dx)))

def process_lateral(annotated, lm, w, h):
    """Handles Craniovertebral Angle (CVA) processing."""
    # Determine which side is facing the camera
    l_vis = lm[mp_pose.PoseLandmark.LEFT_EAR.value].visibility
    r_vis = lm[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility
    
    if l_vis > r_vis:
        ear, sh = lm[mp_pose.PoseLandmark.LEFT_EAR.value], lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        side = "Left Lateral"
    else:
        ear, sh = lm[mp_pose.PoseLandmark.RIGHT_EAR.value], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        side = "Right Lateral"

    ear_px = (int(ear.x * w), int(ear.y * h))
    sh_px = (int(sh.x * w), int(sh.y * h))
    
    # C7 Estimation over clothes
    neck_scale = abs(sh_px[1] - ear_px[1])
    offset = 0.08 if "Left" in side else -0.08
    c7_px = (int(sh_px[0] + (neck_scale * offset)), int(sh_px[1] - (neck_scale * 0.18)))
    
    cva = round(calculate_angle(ear_px, c7_px), 2)
    
    # Rendering
    cv2.line(annotated, c7_px, (c7_px[0] + 100 if "Left" in side else c7_px[0] - 100, c7_px[1]), (200, 200, 200), 2)
    color = (0, 0, 230) if cva < 48 else (0, 165, 255) if cva < 53 else (0, 200, 0)
    cv2.line(annotated, c7_px, ear_px, color, 4, cv2.LINE_AA)
    cv2.circle(annotated, ear_px, 8, (255, 255, 0), -1)
    cv2.circle(annotated, c7_px, 8, (0, 255, 255), -1)
    
    status = "Severe Forward Head" if cva < 48 else "Borderline FHP" if cva < 53 else "Optimal Alignment"
    return annotated, {"Plane": side, "Metric": "CVA", "Value": f"{cva}°", "Status": status}

def process_coronal(annotated, lm, w, h):
    """Handles Head Tilt and Shoulder Level processing."""
    l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
    r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    l_ear_px = (int(l_ear.x * w), int(l_ear.y * h))
    r_ear_px = (int(r_ear.x * w), int(r_ear.y * h))
    l_sh_px = (int(l_sh.x * w), int(l_sh.y * h))
    r_sh_px = (int(r_sh.x * w), int(r_sh.y * h))
    
    head_tilt = round(abs(calculate_angle(l_ear_px, r_ear_px)), 2)
    shoulder_tilt = round(abs(calculate_angle(l_sh_px, r_sh_px)), 2)
    
    # Rendering
    cv2.line(annotated, l_ear_px, r_ear_px, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(annotated, l_sh_px, r_sh_px, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(annotated, l_ear_px, 6, (0, 0, 255), -1)
    cv2.circle(annotated, r_ear_px, 6, (0, 0, 255), -1)
    
    status = "Significant Asymmetry" if head_tilt > 5 or shoulder_tilt > 5 else "Clinically Symmetrical"
    return annotated, {
        "Plane": "Coronal (Frontal)", 
        "Metric": "Head Tilt | Shoulder Drop", 
        "Value": f"{head_tilt}° | {shoulder_tilt}°", 
        "Status": status
    }

def analyze_posture(image_np):
    h, w, _ = image_np.shape
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        return None, "No human pose detected. Please ensure you are visible."
    
    lm = results.pose_landmarks.landmark
    plane = get_plane_type(lm)
    annotated = image_np.copy()
    
    if plane == "Coronal":
        return process_coronal(annotated, lm, w, h), None
    else:
        return process_lateral(annotated, lm, w, h), None

# UI Layout
st.markdown("### 🔬 Clinical Kinematics Engine")
tab1, tab2 = st.tabs(["📁 File Upload", "📷 Mobile Camera"])
input_image = None

with tab1:
    uploaded = st.file_uploader("Upload patient image (Lateral or Coronal)", type=["jpg", "jpeg", "png"])
    if uploaded: input_image = Image.open(uploaded)
with tab2:
    cam_file = st.camera_input("Take patient photo")
    if cam_file: input_image = Image.open(cam_file)

if input_image:
    img_array = np.array(input_image)
    with st.spinner("Executing anatomical routing..."):
        data, err = analyze_posture(img_array)
        
    if err:
        st.error(err)
    else:
        annotated_img, metrics = data
        st.image(annotated_img, channels="RGB", use_container_width=True)
        
        cols = st.columns(3)
        cols[0].metric("Detected Plane", metrics["Plane"])
        cols[1].metric(metrics["Metric"], metrics["Value"])
        cols[2].metric("Diagnostic Status", metrics["Status"])

