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

st.set_page_config(page_title="KINEMA-AI | Clinical", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border-left: 5px solid #3B82F6 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    div[data-testid="stMetricLabel"] > div { color: #94A3B8 !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"] > div { color: #F8FAFC !important; font-weight: 800 !important; }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "patient_db" not in st.session_state:
    st.session_state.patient_db = []

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

def calculate_horizontal_angle(pt1, pt2):
    dx = pt1[0] - pt2[0]
    dy = pt2[1] - pt1[1] 
    return math.degrees(math.atan2(dy, abs(dx)))

def analyze_frame(image_np, strictness):
    res = pose.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None, "Subject not detected. Ensure the photo is saved upright in your gallery before uploading."
        
    annotated = image_np.copy()
    h, w, _ = annotated.shape
    lm = res.pose_landmarks.landmark
    
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
    r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
    
    is_coronal = abs(l_sh.x - r_sh.x) > 0.15 and (l_ear.visibility > 0.6 and r_ear.visibility > 0.6)
    
    if is_coronal:
        l_ear_px, r_ear_px = (int(l_ear.x * w), int(l_ear.y * h)), (int(r_ear.x * w), int(r_ear.y * h))
        l_sh_px, r_sh_px = (int(l_sh.x * w), int(l_sh.y * h)), (int(r_sh.x * w), int(r_sh.y * h))
        
        head_tilt = round(abs(calculate_horizontal_angle(l_ear_px, r_ear_px)), 2)
        cv2.line(annotated, l_ear_px, r_ear_px, (255, 150, 0), 4, cv2.LINE_AA)
        cv2.line(annotated, l_sh_px, r_sh_px, (0, 200, 255), 4, cv2.LINE_AA)
        
        status = "Clinically Symmetrical" if head_tilt < strictness else "Asymmetry Detected"
        metrics = {
            "Assessment Plane": "Coronal (Frontal)", 
            "Primary Metric": "Head Tilt", 
            "Calculated Value": f"{head_tilt}°", 
            "Diagnostic Status": status
        }
        return annotated, metrics
    
    else:
        side = "Left Lateral" if l_ear.visibility > r_ear.visibility else "Right Lateral"
        ear_px = (int(l_ear.x * w), int(l_ear.y * h)) if "Left" in side else (int(r_ear.x * w), int(r_ear.y * h))
        sh_px = (int(l_sh.x * w), int(l_sh.y * h)) if "Left" in side else (int(r_sh.x * w), int(r_sh.y * h))
        
        # Calculate Tragus-Acromion Angle directly
        angle = round(calculate_horizontal_angle(ear_px, sh_px), 2)
        color = (0, 220, 0) if angle >= 50 else (0, 165, 255) if angle >= 45 else (0, 0, 230)
        
        # Draw Horizontal Baseline from Acromion
        cv2.line(annotated, (0, sh_px[1]), (w, sh_px[1]), (100, 100, 100), 1, cv2.LINE_AA) 
        cv2.line(annotated, sh_px, (sh_px[0] + (200 if "Left" in side else -200), sh_px[1]), (255, 255, 0), 3, cv2.LINE_AA) 
        
        # Draw Vector to Tragus
        cv2.line(annotated, sh_px, ear_px, color, 4, cv2.LINE_AA)
        cv2.circle(annotated, ear_px, 7, (255, 255, 255), -1)
        cv2.circle(annotated, sh_px, 7, (255, 255, 0), -1)
        
        cv2.putText(annotated, "Acromion", (sh_px[0] + 15, sh_px[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated, f"Angle: {angle} deg", (ear_px[0] - 50, ear_px[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        status = "Optimal Alignment" if angle >= 50 else "Severe FHP" if angle < 40 else "FHP"
        metrics = {
            "Assessment Plane": f"Sagittal ({side})", 
            "Primary Metric": "Tragus-Acromion Angle", 
            "Calculated Value": f"{angle}°", 
            "Tragus Coordinate": f"{ear_px}",
            "Acromion Coordinate": f"{sh_px}",
            "Diagnostic Status": status
        }
        return annotated, metrics

def generate_pdf(patient_id, metrics, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(37, 99, 235) 
    pdf.rect(0, 0, 210, 25, 'F')
    pdf.set_y(8)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "KINEMA-AI: Clinical Biomechanics Report", ln=True, align="C")
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(30)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(40, 8, "Patient ID:", 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(60, 8, str(patient_id), 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(40, 8, "Date:", 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, str(datetime.date.today()), 0, 1)
    
    pdf.line(10, 40, 200, 40)
    pdf.ln(5)
    pdf.image(image_path, w=140, x=35) 
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Quantitative Kinematic Baseline", ln=True)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(85, 10, "Metric Parameter", border=1, fill=True)
    pdf.cell(105, 10, "Recorded Value", border=1, ln=True, fill=True)
    
    pdf.set_font("Helvetica", "", 10)
    for k, v in metrics.items():
        pdf.cell(85, 10, str(k), border=1)
        pdf.cell(105, 10, str(v), border=1, ln=True)
    return pdf.output()

st.sidebar.markdown("### 👨‍⚕️ Developer Profile")
st.sidebar.markdown("**Aryan Shetty**\n\n*Clinical Physiotherapist & Biomechanics Engineer*")
st.sidebar.caption("Bridging evidence-based rehabilitation with computational kinematics.")
st.sidebar.markdown("[🔗 GitHub Portfolio](#)")
st.sidebar.divider()
st.session_state.strictness = st.sidebar.slider("Coronal Tolerance (°)", 1.0, 10.0, 4.0, 0.5)

st.title("🧬 KINEMA-AI: Markerless Kinematics Engine")
st.markdown("Automated sagittal/coronal recognition using direct anatomical landmarks (Tragus & Acromion) for maximum precision.")

tab1, tab2 = st.tabs(["📸 Static Image Analysis", "🗂️ Jamovi Data Engine"])

with tab1:
    col_input, col_results = st.columns([1, 1.5], gap="large")
    with col_input:
        patient_id = st.text_input("Subject/Patient ID", "SUBJ-001")
        img_file = st.file_uploader("Upload Patient Scan (PNG/JPG)", type=["png", "jpg", "jpeg"])
        st.caption("Note: Ensure the photo is rotated upright in your gallery before uploading.")
    
    with col_results:
        if img_file:
            img_np = np.array(ImageOps.exif_transpose(Image.open(img_file)))
            with st.spinner("Analyzing anatomical routing..."):
                res = analyze_frame(img_np, st.session_state.strictness)
                
            if res[0] is not None:
                frame, metrics = res
                st.image(frame, channels="RGB", use_container_width=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Assessment Plane", metrics["Assessment Plane"])
                m2.metric(metrics["Primary Metric"], metrics["Calculated Value"])
                m3.metric("Diagnostic Status", metrics["Diagnostic Status"])
                
                action1, action2 = st.columns(2)
                with action1:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(tmp.name)
                        pdf_bytes = generate_pdf(patient_id, metrics, tmp.name)
                    st.download_button("📄 Export PDF", data=bytes(pdf_bytes), file_name=f"{patient_id}_Report.pdf", mime="application/pdf", use_container_width=True)
                with action2:
                    if st.button("💾 Save to Jamovi", use_container_width=True):
                        st.session_state.patient_db.append({"Subject_ID": patient_id, "Plane": metrics["Assessment Plane"], "Value": metrics["Calculated Value"], "Classification": metrics["Diagnostic Status"]})
                        st.toast("Record saved to Jamovi Engine!")
            else:
                st.error(res[1])

with tab2:
    if len(st.session_state.patient_db) > 0:
        df = pd.DataFrame(st.session_state.patient_db)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download .CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="Jamovi_Kinematic_Data.csv", mime="text/csv")
    else:
        st.info("No clinical scans saved in this session yet.")
