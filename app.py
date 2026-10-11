import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
from PIL import Image

# Page configuration for compact mobile-first responsive UI
st.set_page_config(
    page_title="CVA Clinical Vision",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,               # High accuracy for clothing occlusion
    enable_segmentation=False,
    min_detection_confidence=0.65
)
mp_drawing = mp.solutions.drawing_utils

def estimate_c7_landmark(shoulder, ear, side="left"):
    """
    Over-clothing compensation:
    Interpolates C7 by projecting medial-superiorly from the shoulder joint
    towards the base of the cervical spine.
    """
    # Shoulder-to-ear vertical distance provides anatomical scale
    neck_scale = abs(shoulder[1] - ear[1])
    
    # Offset slightly posterior and superior from mid-shoulder
    c7_y = shoulder[1] - (neck_scale * 0.18)
    if side == "left":
        c7_x = shoulder[0] + (neck_scale * 0.08)
    else:
        c7_x = shoulder[0] - (neck_scale * 0.08)
        
    return (c7_x, c7_y)

def compute_cva(tragus_pt, c7_pt):
    """
    Computes Craniovertebral Angle (CVA) relative to horizontal baseline.
    tragus_pt, c7_pt: (x, y) in pixel coordinates.
    """
    dx = tragus_pt[0] - c7_pt[0]
    dy = c7_pt[1] - tragus_pt[1]  # Inverted Y axis in pixel coordinates
    
    # Calculate angle relative to horizontal
    cva_rad = math.atan2(dy, abs(dx))
    cva_deg = math.degrees(cva_rad)
    return round(cva_deg, 2)

def analyze_posture(image_np):
    h, w, _ = image_np.shape
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        return None, "No human pose detected. Ensure full lateral view is visible."
    
    lm = results.pose_landmarks.landmark
    
    # Determine lateral orientation (check landmark visibility)
    left_vis = (lm[mp_pose.PoseLandmark.LEFT_EAR.value].visibility + 
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility) / 2
    right_vis = (lm[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility + 
                 lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility) / 2
    
    if left_vis > right_vis and left_vis > 0.5:
        ear_raw = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
        sh_raw = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        side = "left"
    elif right_vis > 0.5:
        ear_raw = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
        sh_raw = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        side = "right"
    else:
        return None, "Landmark visibility too low. Ensure adequate lateral profile lighting."

    # Normalized to pixel coords
    ear_px = (int(ear_raw.x * w), int(ear_raw.y * h))
    sh_px = (int(sh_raw.x * w), int(sh_raw.y * h))
    
    # Anatomical projection for C7 over clothes
    c7_px = (
        int(estimate_c7_landmark(sh_px, ear_px, side)[0]),
        int(estimate_c7_landmark(sh_px, ear_px, side)[1])
    )
    
    # Compute CVA
    cva = compute_cva(ear_px, c7_px)
    
    # Render Overlay
    annotated = image_np.copy()
    
    # 1. Horizontal reference baseline at C7 level
    line_length = int(w * 0.25)
    baseline_end = (c7_px[0] + line_length if side == "left" else c7_px[0] - line_length, c7_px[1])
    cv2.line(annotated, c7_px, baseline_end, (200, 200, 200), 2, cv2.LINE_AA)
    
    # 2. C7 to Tragus vector
    if cva < 48.0:
        vector_color = (0, 0, 230)      # Red: Excessive Forward Head Posture
        status = "Severe Forward Head Posture"
    elif cva < 53.0:
        vector_color = (0, 165, 255)    # Orange: Borderline FHP
        status = "Borderline Forward Posture"
    else:
        vector_color = (0, 200, 0)      # Green: Optimal CVA
        status = "Normal Postural Alignment"
        
    cv2.line(annotated, c7_px, ear_px, vector_color, 4, cv2.LINE_AA)
    cv2.circle(annotated, ear_px, 8, (255, 255, 0), -1)
    cv2.circle(annotated, c7_px, 8, (0, 255, 255), -1)
    
    # 3. Angle arc/text display
    cv2.putText(annotated, f"CVA: {cva} deg", (ear_px[0] - 40, ear_px[1] - 25), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return {
        "annotated_image": annotated,
        "cva": cva,
        "status": status,
        "side": side,
        "c7_coord": c7_px,
        "tragus_coord": ear_px
    }, None

# UI Layout
st.markdown("<h3 style='text-align: center;'>🔬 Markerless Craniovertebral Angle (CVA)</h3>", unsafe_allow_html=True)
st.caption("Cross-platform lateral kinematic posture evaluation engine.")

tab1, tab2 = st.tabs(["📷 Mobile Camera", "📁 File Upload"])

input_image = None

with tab1:
    cam_file = st.camera_input("Take a lateral profile photo")
    if cam_file:
        input_image = Image.open(cam_file)

with tab2:
    uploaded = st.file_uploader("Upload lateral image (PNG/JPG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        input_image = Image.open(uploaded)

if input_image is not None:
    img_array = np.array(input_image)
    with st.spinner("Analyzing anatomical kinematics..."):
        data, err = analyze_posture(img_array)
        
    if err:
        st.error(err)
    else:
        st.image(data["annotated_image"], channels="RGB", use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.metric("Calculated CVA", f"{data['cva']}°")
        c2.metric("Profile Tracked", data["side"].capitalize())
        
        if data["cva"] < 48.0:
            st.error(f"**Diagnostic Status:** {data['status']} (< 48°)")
        elif data["cva"] < 53.0:
            st.warning(f"**Diagnostic Status:** {data['status']} (48° - 53°)")
        else:
            st.success(f"**Diagnostic Status:** {data['status']} (≥ 53°)")
            
        st.info("💡 Data point ready for export into Jamovi or SPSS.")
