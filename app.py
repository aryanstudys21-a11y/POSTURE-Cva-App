import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import pandas as pd
from PIL import Image
from fpdf import FPDF
import tempfile
import datetime

# --- Theme & Visual Scaffolding ---
st.set_page_config(
    page_title="KINEMA-AI | Markerless Biomechanics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .badge-pass { color: #00e676; font-weight: bold; }
    .badge-warn { color: #ffab00; font-weight: bold; }
    .badge-crit { color: #ff1744; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.65
)

# --- Mathematical Engines ---
def calculate_horizontal_angle(pt1, pt2):
    """Calculates planar angle relative to horizontal baseline (degrees)."""
    dx = pt1[0] - pt2[0]
    dy = pt2[1] - pt1[1]  # Inverted screen coordinates
    return math.degrees(math.atan2(dy, abs(dx)))

def estimate_c7(sh_px, ear_px, side):
    """
    Kinematic prior for C7 under clothing:
    Projects posterior-superior vector based on vertical cranio-cervical distance.
    """
    scale = abs(sh_px[1] - ear_px[1])
    offset_x = 0.08 if "Left" in side else -0.08
    c7_x = int(sh_px[0] + (scale * offset_x))
    c7_y = int(sh_px[1] - (scale * 0.18))
    return (c7_x, c7_y)

def generate_pdf_report(patient_name, pid, metrics, annotated_img_path):
    """Compiles a clinical kinematics summary into a downloadable PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "KINEMA-AI: Clinical Biomechanics Report", ln=True, align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Date: {datetime.date.today()} | Patient ID: {pid} | Name: {patient_name}", ln=True, align="C")
    pdf.ln(5)
    
    # Save Image to PDF
    pdf.image(annotated_img_path, x=35, y=35, w=140)
    pdf.set_y(185)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Kinematic Assessments & Normative Comparison:", ln=True)
    pdf.set_font("Helvetica", size=10)
    
    for k, v in metrics.items():
        pdf.cell(0, 6, f"- {k}: {v}", ln=True)
        
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Disclaimer: Generated via markerless computational computer vision. Formatted for research correlation matrices and clinical documentation.")
    return pdf.output()

# --- Kinematic Routing Pipeline ---
def analyze_frame(image_np):
    h, w, _ = image_np.shape
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    
    if not res.pose_landmarks:
        return None, "No human pose landmarks detected. Check framing."
    
    lm = res.pose_landmarks.landmark
    annotated = image_np.copy()
    
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_span = abs(l_sh.x - r_sh.x)
    
    # Plane classification threshold
    is_coronal = shoulder_span > 0.13
    
    if is_coronal:
        l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
        
        l_ear_px = (int(l_ear.x * w), int(l_ear.y * h))
        r_ear_px = (int(r_ear.x * w), int(r_ear.y * h))
        l_sh_px = (int(l_sh.x * w), int(l_sh.y * h))
        r_sh_px = (int(r_sh.x * w), int(r_sh.y * h))
        
        head_tilt = round(abs(calculate_horizontal_angle(l_ear_px, r_ear_px)), 2)
        sh_tilt = round(abs(calculate_horizontal_angle(l_sh_px, r_sh_px)), 2)
        
        # Clinical Overlays
        cv2.line(annotated, l_ear_px, r_ear_px, (0, 240, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, l_sh_px, r_sh_px, (255, 180, 0), 3, cv2.LINE_AA)
        cv2.circle(annotated, l_ear_px, 6, (0, 0, 255), -1)
        cv2.circle(annotated, r_ear_px, 6, (0, 0, 255), -1)
        cv2.circle(annotated, l_sh_px, 6, (0, 255, 0), -1)
        cv2.circle(annotated, r_sh_px, 6, (0, 255, 0), -1)
        
        status = "Clinically Symmetrical" if (head_tilt < 4.0 and sh_tilt < 4.0) else "Postural Asymmetry Observed"
        
        data_bundle = {
            "plane": "Coronal (Frontal)",
            "primary_metric": "Head Tilt",
            "primary_val": f"{head_tilt}°",
            "sec_metric": "Acromial Tilt",
            "sec_val": f"{sh_tilt}°",
            "normative": "Normal: < 4.0°",
            "status": status,
            "raw": {"Head_Tilt_Deg": head_tilt, "Shoulder_Tilt_Deg": sh_tilt}
        }
    else:
        l_vis = lm[mp_pose.PoseLandmark.LEFT_EAR.value].visibility
        r_vis = lm[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility
        side = "Left Lateral" if l_vis > r_vis else "Right Lateral"
        
        ear_raw = lm[mp_pose.PoseLandmark.LEFT_EAR.value] if "Left" in side else lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
        sh_raw = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value] if "Left" in side else lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        ear_px = (int(ear_raw.x * w), int(ear_raw.y * h))
        sh_px = (int(sh_raw.x * w), int(sh_raw.y * h))
        c7_px = estimate_c7(sh_px, ear_px, side)
        
        cva = round(calculate_horizontal_angle(ear_px, c7_px), 2)
        
        # Clinical vector visuals
        baseline_dx = 120 if "Left" in side else -120
        cv2.line(annotated, c7_px, (c7_px[0] + baseline_dx, c7_px[1]), (210, 210, 210), 2, cv2.LINE_AA)
        
        color = (0, 0, 230) if cva < 48 else (0, 165, 255) if cva < 53 else (0, 220, 0)
        cv2.line(annotated, c7_px, ear_px, color, 4, cv2.LINE_AA)
        cv2.circle(annotated, ear_px, 7, (255, 255, 0), -1)
        cv2.circle(annotated, c7_px, 7, (0, 255, 255), -1)
        
        status = "Severe Forward Head Posture" if cva < 48 else "Borderline Alignment" if cva < 53 else "Optimal Cervical Alignment"
        
        data_bundle = {
            "plane": f"Sagittal ({side})",
            "primary_metric": "Craniovertebral Angle (CVA)",
            "primary_val": f"{cva}°",
            "sec_metric": "Thoracic Reference Offset",
            "sec_val": f"{round(abs(sh_px[0] - c7_px[0]), 1)} px",
            "normative": "Optimal: 50° - 55°",
            "status": status,
            "raw": {"CVA_Deg": cva, "View": side}
        }
        
    return annotated, data_bundle

# --- Application Layout ---
st.sidebar.title("🧬 KINEMA-AI Core")
st.sidebar.caption("Markerless Clinical Biomechanics Engine")

patient_name = st.sidebar.text_input("Patient / Subject Name", "Aryan Shetty")
patient_id = st.sidebar.text_input("Trial ID (Jamovi Reference)", "SUBJ-093")
view_mode = st.sidebar.radio("Navigation", ["🔬 Live Kinematic Analysis", "📊 Research Batch Data (Jamovi)"])

if view_mode == "🔬 Live Kinematic Analysis":
    st.title("Markerless Postural Kinematics")
    st.markdown("Automated sagittal/coronal recognition with clothing compensation.")
    
    input_tab, cam_tab = st.tabs(["📁 Image / Gallery Upload", "📷 Live Camera Input"])
    selected_image = None
    
    with input_tab:
        file = st.file_uploader("Upload patient profile (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if file:
            selected_image = Image.open(file)
            
    with cam_tab:
        cam_shot = st.camera_input("Capture profile directly")
        if cam_shot:
            selected_image = Image.open(cam_shot)
            
    if selected_image:
        img_np = np.array(selected_image)
        with st.spinner("Processing anatomical landmarks & plane projection..."):
            annotated_frame, result = analyze_frame(img_np)
            
        if not result or isinstance(result, str):
            st.error(result)
        else:
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                st.image(annotated_frame, channels="RGB", use_container_width=True)
                
            with col2:
                st.subheader("Diagnostic Metrics")
                st.metric("Detected Assessment Plane", result["plane"])
                
                m1, m2 = st.columns(2)
                m1.metric(result["primary_metric"], result["primary_val"])
                m2.metric(result["sec_metric"], result["sec_val"])
                
                st.markdown(f"**Normative Reference:** `{result['normative']}`")
                
                if "Optimal" in result["status"] or "Symmetrical" in result["status"]:
                    st.success(f"**Finding:** {result['status']}")
                elif "Borderline" in result["status"]:
                    st.warning(f"**Finding:** {result['status']}")
                else:
                    st.error(f"**Finding:** {result['status']}")
                    
                # PDF Generation Action
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    Image.fromarray(annotated_frame).save(tmp.name)
                    pdf_bytes = generate_pdf_report(
                        patient_name,
                        patient_id,
                        {
                            "Plane": result["plane"],
                            result["primary_metric"]: result["primary_val"],
                            result["sec_metric"]: result["sec_val"],
                            "Status": result["status"]
                        },
                        tmp.name
                    )
                    
                st.download_button(
                    label="📄 Export Clinical PDF Report",
                    data=bytes(pdf_bytes),
                    file_name=f"{patient_id}_kinematic_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

elif view_mode == "📊 Research Batch Data (Jamovi)":
    st.title("Research Statistical Export Engine")
    st.markdown("Designed for direct import into statistical packages (Jamovi, SPSS, R)[span_7](start_span)[span_7](end_span)[span_8](start_span)[span_8](end_span).")
    
    mock_df = pd.DataFrame([
        {"Subject_ID": "SUBJ-001", "Plane": "Sagittal", "CVA_Deg": 46.2, "Classification": "FHP", "Pain_VAS": 6},
        {"Subject_ID": "SUBJ-002", "Plane": "Sagittal", "CVA_Deg": 54.1, "Classification": "Normal", "Pain_VAS": 1},
        {"Subject_ID": patient_id, "Plane": "Sagittal", "CVA_Deg": 53.4, "Classification": "Normal", "Pain_VAS": 2},
    ])
    
    st.dataframe(mock_df, use_container_width=True)
    
    csv_data = mock_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset (.CSV for Jamovi/SPSS)",
        data=csv_data,
        file_name="kinematic_posture_dataset.csv",
        mime="text/csv"
    )
