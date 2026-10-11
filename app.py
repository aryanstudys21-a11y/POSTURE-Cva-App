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

# --- 2. Vision Pipeline ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)

def calculate_horizontal_angle(pt1, pt2):
    dx = pt1[0] - pt2[0]
    dy = pt2[1] - pt1[1] 
    return math.degrees(math.atan2(dy, abs(dx)))

def auto_orient_image(image_np):
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    for rot in rotations:
        test_img = cv2.rotate(image_np, rot) if rot is not None else image_np
        res = pose.process(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            nose_y = lm[mp_pose.PoseLandmark.NOSE.value].y
            l_sh_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            r_sh_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            
            # STRICT GRAVITY CHECK: Nose must be definitively higher than shoulders
            if (l_sh_y - nose_y) > 0.10 and (r_sh_y - nose_y) > 0.10:
                return test_img, res.pose_landmarks.landmark
    return image_np, None

def analyze_frame(image_np, strictness):
    oriented_img, lm = auto_orient_image(image_np)
    if not lm:
        return None, "Subject not detected. Ensure head and shoulders are visibly upright in the frame."
        
    annotated = oriented_img.copy()
    h, w, _ = annotated.shape
    
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
            "Left Tragus (X,Y)": f"{l_ear_px}",
            "Right Tragus (X,Y)": f"{r_ear_px}",
            "Diagnostic Status": status
        }
        return annotated, metrics
    
    else:
        side = "Left Lateral" if l_ear.visibility > r_ear.visibility else "Right Lateral"
        ear_px = (int(l_ear.x * w), int(l_ear.y * h)) if "Left" in side else (int(r_ear.x * w), int(r_ear.y * h))
        sh_px = (int(l_sh.x * w), int(l_sh.y * h)) if "Left" in side else (int(r_sh.x * w), int(r_sh.y * h))
        
        scale = abs(sh_px[1] - ear_px[1])
        
        # TRUE POSTERIOR C7 ESTIMATION
        # If Left Lateral (facing right), back is on the left (-x). If Right Lateral, back is on the right (+x).
        offset_x = -0.28 if "Left" in side else 0.28 
        offset_y = -0.05 # Slightly superior to the acromion
        c7_px = (int(sh_px[0] + (scale * offset_x)), int(sh_px[1] + (scale * offset_y)))
        
        cva = round(calculate_horizontal_angle(ear_px, c7_px), 2)
        color = (0, 220, 0) if cva >= 50 else (0, 165, 255) if cva >= 45 else (0, 0, 230)
        
        cv2.line(annotated, (0, c7_px[1]), (w, c7_px[1]), (255, 255, 255), 1, cv2.LINE_AA) 
        cv2.line(annotated, c7_px, (c7_px[0] + (200 if "Left" in side else -200), c7_px[1]), (255, 255, 0), 3, cv2.LINE_AA) 
        cv2.line(annotated, c7_px, ear_px, color, 4, cv2.LINE_AA)
        cv2.circle(annotated, ear_px, 7, (255, 255, 255), -1)
        cv2.circle(annotated, c7_px, 7, (255, 255, 0), -1)
        cv2.putText(annotated, "C7", (c7_px[0] + 15, c7_px[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated, f"CVA: {cva} deg", (ear_px[0] - 50, ear_px[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        status = "Optimal Alignment" if cva >= 50 else "Severe FHP" if cva < 40 else "FHP"
        metrics = {
            "Assessment Plane": f"Sagittal ({side})", 
            "Primary Metric": "True Craniovertebral Angle (CVA)", 
            "Calculated Value": f"{cva}°", 
            "Tragus Coordinate (X,Y)": f"{ear_px}",
            "C7 Vertex Coordinate (X,Y)": f"{c7_px}",
            "Baseline Horizontal": f"Y = {c7_px[1]}",
            "Diagnostic Status": status
        }
        return annotated, metrics

def generate_pdf(patient_id, metrics, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(15, 82, 186) 
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
    pdf.cell(75, 10, "Metric Parameter", border=1, fill=True)
    pdf.cell(115, 10, "Recorded Value", border=1, ln=True, fill=True)
    
    pdf.set_font("Helvetica", "", 10)
    for k, v in metrics.items():
        pdf.cell(75, 10, str(k), border=1)
        pdf.cell(115, 10, str(v), border=1, ln=True)
        
    return pdf.output()

# --- 3. Interactive Multi-Page UI ---
st.sidebar.markdown("### 🧬 KINEMA-AI Clinical")
page = st.sidebar.radio("Navigation", ["📸 Active Scanner", "🗂️ Jamovi Data Engine", "⚙️ Engine Settings"])

if page == "📸 Active Scanner":
    st.markdown("## 🔬 Markerless Postural Kinematics")
    st.markdown("Automated sagittal/coronal recognition with true C7 coordinate estimation.")
    
    patient_id = st.text_input("Subject/Patient ID", "SUBJ-001")
    img_file = st.file_uploader("Upload Patient Scan (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if img_file:
        img_np = np.array(ImageOps.exif_transpose(Image.open(img_file)))
        strictness = st.session_state.get("strictness", 4.0)
        
        with st.spinner("Executing strict gravity validation and anatomical routing..."):
            res = analyze_frame(img_np, strictness)
            
        if res[0] is not None:
            frame, metrics = res
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.image(frame, channels="RGB", use_container_width=True)
            with col2:
                st.metric("Detected Assessment Plane", metrics["Assessment Plane"])
                st.metric(metrics["Primary Metric"], metrics["Calculated Value"])
                st.metric("Diagnostic Status", metrics["Diagnostic Status"])
                
                st.markdown("---")
                
                # PDF Generation
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pdf_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    Image.fromarray(pdf_img).save(tmp.name)
                    pdf_bytes = generate_pdf(patient_id, metrics, tmp.name)
                    
                st.download_button("📄 Export Clinical PDF Report", data=bytes(pdf_bytes), file_name=f"{patient_id}_Report.pdf", mime="application/pdf", use_container_width=True)
                
                # Jamovi Database Save
                if st.button("💾 Save to Jamovi Records", use_container_width=True):
                    record = {
                        "Subject_ID": patient_id, 
                        "Plane": metrics["Assessment Plane"], 
                        "Value": metrics["Calculated Value"], 
                        "Classification": metrics["Diagnostic Status"]
                    }
                    st.session_state.patient_db.append(record)
                    st.success("Successfully saved to Data Engine.")
        else:
            st.error(res[1])

elif page == "🗂️ Jamovi Data Engine":
    st.markdown("## 📊 Statistical Export Engine")
    st.markdown("Records formatted for non-parametric testing in Jamovi/SPSS.")
    if len(st.session_state.patient_db) > 0:
        df = pd.DataFrame(st.session_state.patient_db)
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download .CSV for Jamovi", data=df.to_csv(index=False).encode('utf-8'), file_name="Jamovi_Kinematic_Data.csv", mime="text/csv")
    else:
        st.info("No clinical scans saved in this session yet.")

elif page == "⚙️ Engine Settings":
    st.markdown("## ⚙️ Engine Tuning")
    st.session_state.strictness = st.slider("Coronal Symmetry Tolerance (°)", 1.0, 10.0, 4.0, 0.5)
