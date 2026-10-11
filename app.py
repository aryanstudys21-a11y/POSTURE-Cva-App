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

# --- 1. Clinical UI Overhaul ---
st.set_page_config(page_title="KINEMA-AI | Clinical", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    :root { --primary: #0F52BA; --background: #F8F9FA; --text: #2B2D42; }
    .stApp { background-color: #F4F7F6; color: #1E293B; }
    #MainMenu, footer, header {visibility: hidden;}
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #0F52BA;
    }
</style>
""", unsafe_allow_html=True)

if "patient_db" not in st.session_state:
    st.session_state.patient_db = []

# --- 2. Vision Pipeline ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)

def calculate_horizontal_angle(pt1, pt2):
    dx = pt1[0] - pt2[0]
    dy = pt2[1] - pt1[1] 
    return math.degrees(math.atan2(dy, abs(dx)))

def estimate_c7(sh_px, ear_px, side):
    scale = abs(sh_px[1] - ear_px[1])
    offset_x = 0.08 if "Left" in side else -0.08
    return (int(sh_px[0] + (scale * offset_x)), int(sh_px[1] - (scale * 0.18)))

def draw_goniometer_arc(annotated, center, angle, side, color):
    radius = 50
    start_angle = 180 if "Left" in side else 0
    end_angle = start_angle - angle if "Left" in side else start_angle - angle
    baseline_end = (center[0] + (150 if "Right" in side else -150), center[1])
    cv2.line(annotated, center, baseline_end, (200, 200, 200), 2, cv2.LINE_AA)
    
    axes = (radius, radius)
    start = 0 if "Right" in side else 180
    end = -angle if "Right" in side else 180 - angle
    cv2.ellipse(annotated, center, axes, 0, start, end, color, 2, cv2.LINE_AA)
    
    text_pos = (center[0] + (60 if "Right" in side else -100), center[1] - 30)
    cv2.putText(annotated, f"{round(angle, 1)} deg", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def analyze_frame(image_np, strictness_threshold):
    h, w, _ = image_np.shape
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    
    if not res.pose_landmarks: 
        return None, "Subject not detected. Please ensure at least one ear and shoulder are clearly visible."
    
    lm = res.pose_landmarks.landmark
    annotated = image_np.copy()
    
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_vis = lm[mp_pose.PoseLandmark.LEFT_EAR.value].visibility
    r_vis = lm[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility
    
    if abs(l_sh.x - r_sh.x) > 0.13 and (l_vis > 0.5 and r_vis > 0.5):
        l_ear_px = (int(lm[mp_pose.PoseLandmark.LEFT_EAR.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_EAR.value].y * h))
        r_ear_px = (int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h))
        l_sh_px = (int(l_sh.x * w), int(l_sh.y * h))
        r_sh_px = (int(r_sh.x * w), int(r_sh.y * h))
        
        head_tilt = round(abs(calculate_horizontal_angle(l_ear_px, r_ear_px)), 2)
        sh_tilt = round(abs(calculate_horizontal_angle(l_sh_px, r_sh_px)), 2)
        
        cv2.line(annotated, l_ear_px, r_ear_px, (255, 150, 0), 4, cv2.LINE_AA)
        cv2.line(annotated, l_sh_px, r_sh_px, (0, 200, 255), 4, cv2.LINE_AA)
        
        status = "Clinically Symmetrical" if (head_tilt < strictness_threshold and sh_tilt < strictness_threshold) else "Asymmetry Detected"
        return annotated, {"Plane": "Coronal (Frontal)", "Primary_Metric": "Head Tilt", "Primary_Value": head_tilt, "Status": status}
    
    else:
        side = "Left Lateral" if l_vis > r_vis else "Right Lateral"
        ear_raw = lm[mp_pose.PoseLandmark.LEFT_EAR.value] if "Left" in side else lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
        sh_raw = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value] if "Left" in side else lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        ear_px = (int(ear_raw.x * w), int(ear_raw.y * h))
        sh_px = (int(sh_raw.x * w), int(sh_raw.y * h))
        c7_px = estimate_c7(sh_px, ear_px, side)
        
        cva = round(calculate_horizontal_angle(ear_px, c7_px), 2)
        
        color = (186, 82, 15) if cva >= 50 else (50, 50, 220)
        cv2.line(annotated, c7_px, ear_px, color, 4, cv2.LINE_AA)
        draw_goniometer_arc(annotated, c7_px, cva, side, color)
        
        cv2.circle(annotated, ear_px, 6, (255, 255, 255), -1)
        cv2.circle(annotated, c7_px, 6, (255, 255, 255), -1)
        
        if cva >= 50: status = "Optimal Alignment"
        elif cva >= 45: status = "Mild Forward Head Posture"
        elif cva >= 40: status = "Moderate FHP"
        else: status = "Severe FHP"
            
        return annotated, {"Plane": f"Sagittal ({side})", "Primary_Metric": "Craniovertebral Angle", "Primary_Value": cva, "Status": status}

def generate_pdf(patient_id, metrics, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "KINEMA-AI: Clinical Biomechanics Report", ln=True, align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Date: {datetime.date.today()} | Patient ID: {patient_id}", ln=True, align="C")
    pdf.ln(5)
    pdf.image(image_path, x=35, y=35, w=140)
    pdf.set_y(185)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Kinematic Assessments:", ln=True)
    pdf.set_font("Helvetica", size=10)
    for k, v in metrics.items():
        pdf.cell(0, 6, f"- {k}: {v}", ln=True)
    return pdf.output()

# --- 3. Interactive Multi-Page UI ---
st.sidebar.markdown("### 🧬 KINEMA-AI Clinical")
page = st.sidebar.radio("Navigation", ["📸 Active Scanner", "🗂️ Jamovi Data Engine", "⚙️ Clinical Settings"])

if page == "📸 Active Scanner":
    st.markdown("## Active Kinematic Scanner")
    
    patient_id = st.text_input("Subject/Patient ID", "SUBJ-001")
    img_file = st.file_uploader("Upload Patient Scan (PNG/JPG)", type=["png", "jpg", "jpeg"])
        
    if img_file:
        # THE FIX: Manual Rotation Override Tools
        st.markdown("#### 🔄 Image Orientation Check")
        rotation = st.radio("If the image uploaded sideways, fix it here before processing:", 
                            ["Looks Good (No Change)", "Rotate 90° Right", "Rotate 90° Left", "Rotate 180°"], 
                            horizontal=True)
        
        img_pil = ImageOps.exif_transpose(Image.open(img_file))
        
        if rotation == "Rotate 90° Right": img_pil = img_pil.rotate(-90, expand=True)
        elif rotation == "Rotate 90° Left": img_pil = img_pil.rotate(90, expand=True)
        elif rotation == "Rotate 180°": img_pil = img_pil.rotate(180, expand=True)
            
        img_np = np.array(img_pil)
        strictness = st.session_state.get("strictness", 4.0)
        
        with st.spinner("Calculating biomechanical vectors..."):
            res = analyze_frame(img_np, strictness)
            
        if res[0] is not None:
            frame, metrics = res
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.image(frame, channels="RGB", use_container_width=True)
            with col2:
                st.metric("Detected Plane", metrics["Plane"])
                st.metric(metrics["Primary_Metric"], f"{metrics['Primary_Value']}°")
                st.metric("Diagnostic Status", metrics["Status"])
                
                st.markdown("---")
                # Automated PDF Engine
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    Image.fromarray(frame).save(tmp.name)
                    pdf_bytes = generate_pdf(patient_id, metrics, tmp.name)
                    
                st.download_button("📄 Export Clinical PDF", data=bytes(pdf_bytes), file_name=f"{patient_id}_Report.pdf", mime="application/pdf", use_container_width=True)
                
                if st.button("💾 Save to Jamovi Records", use_container_width=True):
                    record = {"Subject_ID": patient_id, "Plane": metrics["Plane"], "Angle_Deg": metrics["Primary_Value"], "Classification": metrics["Status"]}
                    st.session_state.patient_db.append(record)
                    st.success("Saved to Data Engine.")
        else:
            st.error(res[1])

elif page == "🗂️ Jamovi Data Engine":
    st.markdown("## Statistical Export Engine")
    st.markdown("Records formatted for descriptive statistics and non-parametric testing.")
    if len(st.session_state.patient_db) > 0:
        df = pd.DataFrame(st.session_state.patient_db)
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download .CSV for Jamovi/SPSS", data=df.to_csv(index=False).encode('utf-8'), file_name="Jamovi_Kinematic_Data.csv", mime="text/csv")
    else:
        st.info("No clinical scans saved.")

elif page == "⚙️ Clinical Settings":
    st.markdown("## Engine Tuning")
    st.session_state.strictness = st.slider("Coronal Symmetry Tolerance (°)", 1.0, 10.0, 4.0, 0.5)
