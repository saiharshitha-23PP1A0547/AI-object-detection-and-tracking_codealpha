"""
AI Object Detection and Tracking System
CodeAlpha Artificial Intelligence Internship
Author: CodeAlpha Intern
Technologies: Python 3.10+, OpenCV, Ultralytics YOLO, ByteTrack / SORT, Streamlit
"""

import os
import sys
import time
import tempfile
import subprocess
import numpy as np
import cv2
import streamlit as st

# Import custom modular components
from detector import YOLODetector
from tracker import ByteTracker

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Object Detection and Tracking | CodeAlpha",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; text-align: center; }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #0284C7; }
    .metric-lbl { font-size: 0.85rem; color: #64748B; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

TRACK_COLORS = [
    (56, 189, 248), (52, 211, 153), (251, 191, 36), (244, 114, 182),
    (167, 139, 250), (251, 146, 60), (74, 222, 128), (239, 68, 68),
    (34, 211, 238), (245, 158, 11), (147, 51, 234), (236, 72, 153),
]

def get_track_color(track_id: int):
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]

@st.cache_resource(show_spinner=False)
def load_detector(model_name: str):
    return YOLODetector(model_name=model_name)

def convert_video_to_h264(input_path: str, output_path: str) -> bool:
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast", output_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"[FFmpeg Warning] Re-encoding skipped: {e}")
        return False

def main():
    st.markdown('<div class="main-header">🎯 AI Object Detection and Tracking System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CodeAlpha Artificial Intelligence Internship Project &bull; Real-time Computer Vision & Multi-Object Tracking</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        input_source = st.radio("📹 Select Video Input Source", ["Upload Video File", "Live Webcam", "Built-in Sample Video"], index=0)
        st.divider()
        st.subheader("🧠 Model & Inference")
        model_choice = st.selectbox("YOLO Pretrained Model", ["yolo11n.pt", "yolov8n.pt", "yolo11s.pt"], index=0)
        conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
        iou_threshold = st.slider("NMS / IoU Threshold", 0.10, 0.80, 0.45, 0.05)
        st.divider()
        st.subheader("🔄 Tracking Engine")
        tracker_type = st.selectbox("Tracking Algorithm", ["ByteTrack (Recommended)", "Classic SORT"], index=0)
        max_age = st.slider("Max Age (Occlusion Tolerance)", 5, 60, 30, 5)
        min_hits = st.slider("Min Hits to Confirm Track", 1, 5, 2, 1)
        st.divider()
        st.subheader("🎨 Display Options")
        show_trajectories = st.checkbox("Draw Movement Trails", value=True)
        show_track_id = st.checkbox("Draw Tracking IDs", value=True)
        show_confidence = st.checkbox("Show Detection Confidence %", value=True)
        box_thickness = st.slider("Bounding Box Thickness", 1, 4, 2)

    with st.spinner("Initializing YOLO model... (First-time download may take ~10-20 seconds)"):
        try:
            detector = load_detector(model_choice)
        except Exception as e:
            st.error(f"Failed to load YOLO model: {e}")
            st.stop()

    all_classes = list(detector.classes.values()) if detector.classes else ["person", "car", "bus", "truck", "motorcycle", "bicycle", "dog"]
    default_selected = [c for c in ["person", "car", "bus", "truck", "motorcycle", "bicycle", "dog"] if c in all_classes]

    with st.sidebar:
        st.subheader("🏷️ Object Class Filters")
        selected_classes = st.multiselect("Filter visible object classes:", options=all_classes, default=default_selected)

    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "processed_video_path" not in st.session_state:
        st.session_state.processed_video_path = None

    col_main, col_stats = st.columns([2.2, 1.0])
    video_source_path = None
    is_webcam = False

    with col_main:
        if input_source == "Upload Video File":
            uploaded_file = st.file_uploader("📁 Upload video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_source_path = tfile.name
                st.success(f"Loaded: `{uploaded_file.name}` ({round(uploaded_file.size / 1024 / 1024, 2)} MB)")
            else:
                st.info("💡 Upload a video file above, or select 'Built-in Sample Video' in the sidebar.")
        elif input_source == "Live Webcam":
            is_webcam = True
            st.warning("⚠️ **Webcam Note:** Direct hardware access requires local camera device. For remote/cloud hosts, use Upload Video.")
            video_source_path = 0
        elif input_source == "Built-in Sample Video":
            sample_path = "sample_traffic.mp4"
            if not os.path.exists(sample_path):
                from create_sample_video import generate_synthetic_sample_video
                generate_synthetic_sample_video(sample_path, num_frames=150)
            video_source_path = sample_path
            st.success("Loaded built-in benchmark video: `sample_traffic.mp4`")

        btn_c1, btn_c2, btn_c3 = st.columns(3)
        if btn_c1.button("▶️ Start Processing", type="primary", use_container_width=True):
            st.session_state.is_running = True
        if btn_c2.button("⏹️ Stop", use_container_width=True):
            st.session_state.is_running = False
        if btn_c3.button("🔄 Reset System", use_container_width=True):
            st.session_state.is_running = False
            st.session_state.processed_video_path = None
            st.rerun()

        frame_placeholder = st.empty()
        progress_bar = st.progress(0.0)

    with col_stats:
        st.subheader("📊 Live Telemetry")
        m1 = st.empty()
        m2 = st.empty()
        m3 = st.empty()
        m4 = st.empty()
        st.subheader("📈 Class Breakdown")
        class_table_placeholder = st.empty()
        st.subheader("💾 Export Deliverable")
        download_placeholder = st.empty()

    if st.session_state.is_running and video_source_path is not None:
        cap = cv2.VideoCapture(video_source_path)
        if not cap.isOpened():
            st.error(f"Error opening video source '{video_source_path}'.")
            st.session_state.is_running = False
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs("output", exist_ok=True)
        raw_output_path = os.path.join("output", "temp_annotated.mp4")
        final_output_path = os.path.join("output", "annotated_tracking_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(raw_output_path, fourcc, fps_in, (width, height))

        tracker = ByteTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        frame_idx = 0
        observed_ids = set()
        fps_history = []

        try:
            while st.session_state.is_running:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1

                detections = detector.detect(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold, allowed_classes=selected_classes)
                tracked_objects = tracker.update(detections)

                for obj in tracked_objects:
                    observed_ids.add(obj["track_id"])

                annotated_frame = frame.copy()
                if show_trajectories:
                    for obj in tracked_objects:
                        traj = obj.get("trajectory", [])
                        t_color = get_track_color(obj["track_id"])
                        for p in range(1, len(traj)):
                            cv2.line(annotated_frame, (int(traj[p-1][0]), int(traj[p-1][1])), (int(traj[p][0]), int(traj[p][1])), t_color, thickness=box_thickness)

                for obj in tracked_objects:
                    x1, y1, x2, y2 = [int(v) for v in obj["box"]]
                    color = get_track_color(obj["track_id"])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=box_thickness)
                    label = f"ID #{obj['track_id']} | {obj['class_name']} ({int(obj['score']*100)}%)"
                    cv2.putText(annotated_frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                out_writer.write(annotated_frame)
                fps_history.append(1.0 / max(1e-4, time.time() - t0))
                if len(fps_history) > 30: fps_history.pop(0)

                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                m1.markdown(f'<div class="metric-card"><div class="metric-val">{len(tracked_objects)}</div><div class="metric-lbl">Objects Currently Detected</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-card"><div class="metric-val">{len(observed_ids)}</div><div class="metric-lbl">Unique Tracked IDs</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-card"><div class="metric-val">{round(np.mean(fps_history), 1)}</div><div class="metric-lbl">Processing FPS</div></div>', unsafe_allow_html=True)
                m4.markdown(f'<div class="metric-card"><div class="metric-val">{frame_idx}</div><div class="metric-lbl">Frames Processed</div></div>', unsafe_allow_html=True)
        finally:
            cap.release()
            out_writer.release()

        st.session_state.is_running = False
        convert_video_to_h264(raw_output_path, final_output_path)
        st.session_state.processed_video_path = final_output_path if os.path.exists(final_output_path) else raw_output_path
        st.success(f"Processing complete! {len(observed_ids)} unique tracked objects.")

    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
        with open(st.session_state.processed_video_path, "rb") as vf:
            download_placeholder.download_button("📥 Download Annotated Video (MP4)", vf.read(), file_name="annotated_tracking.mp4", mime="video/mp4", use_container_width=True)

if __name__ == "__main__":
    main()
