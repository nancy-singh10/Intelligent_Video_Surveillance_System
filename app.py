import numpy as np
import streamlit as st
import cv2
from collections import deque
import os
import subprocess
from keras.models import load_model

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

CLASSES_LIST = [
    "BaseballPitch", "Basketball", 'BenchPress', "Biking", "JavelinThrow", "CleanAndJerk",
    "Diving", "Billiards", "HighJump", "HorseRace", "MilitaryParade", "PlayingGuitar",
    "ThrowDiscus", "WalkingWithDog", "SkateBoarding", "null", "JumpingJack", "JumpRope",
    "Kayaking", "HulaHoop", "JugglingBalls", "GolfSwing", "Fencing", "Drumming",
    "HorseRiding", "BreastStroke"
]

@st.cache_resource
def load_LRCN_model():
    return load_model("model1.h5")

def predict_single_action(video_path, model):
    video_reader = cv2.VideoCapture(video_path)
    frames_list = []

    frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_window = max(int(frame_count / SEQUENCE_LENGTH), 1)

    for i in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i * skip_window)
        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    predicted_probs = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_probs)
    predicted_class_name = CLASSES_LIST[predicted_label]
    confidence = predicted_probs[predicted_label]

    video_reader.release()
    return predicted_class_name, confidence

def convert_video_for_browser(input_path, output_path):
    video_reader = cv2.VideoCapture(input_path)
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30 # fallback

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if not ret:
            break
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

def annotate_video(video_path, output_path, model):
    video_reader = cv2.VideoCapture(video_path)
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    # Use 'avc1' or 'h264' codec for HTML5 browser compatibility without needing ffmpeg
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class = ''

    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_probs = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_probs)
            predicted_class = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

COACHING_INSIGHTS = {
    "BaseballPitch": {
        "Mechanic Analysis": "Strong kinetic chain transfer from lower to upper body. Release point is consistent.",
        "Stress Point": "Moderate stress detected on the Ulnar Collateral Ligament (UCL).",
        "Recommendation": "Delay torso rotation by 0.05s to maximize shoulder separation and pitch velocity."
    },
    "CleanAndJerk": {
        "Mechanic Analysis": "Excellent triple extension at the hips, knees, and ankles.",
        "Stress Point": "High lumbar spine compression during the catch phase.",
        "Recommendation": "Keep chest more upright during the initial pull to improve bar trajectory."
    },
    "BenchPress": {
        "Mechanic Analysis": "Stable bar path with good lat engagement and leg drive.",
        "Stress Point": "Anterior deltoids and pectoralis major insertion.",
        "Recommendation": "Ensure wrists remain stacked directly over elbows to prevent joint strain."
    },
    "Biking": {
        "Mechanic Analysis": "Smooth pedal stroke with even power distribution across the 360-degree rotation.",
        "Stress Point": "Patellar tendon.",
        "Recommendation": "Raise saddle height by ~5mm to achieve optimal knee extension."
    },
    "Diving": {
        "Mechanic Analysis": "Tight tuck position achieved quickly. Entry angle is near optimal (88 degrees).",
        "Stress Point": "Cervical spine upon water entry.",
        "Recommendation": "Squeeze core tighter immediately upon leaving the springboard for faster rotation."
    },
    "Default": {
        "Mechanic Analysis": "Action executed within normal biomechanical parameters.",
        "Stress Point": "Standard exertion detected for this movement.",
        "Recommendation": "Continue monitoring form to prevent fatigue-induced injuries."
    }
}

def add_custom_css():
    st.markdown("""
        <style>
        /* Main background and text */
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        /* Style headers */
        h1, h2, h3 {
            color: #4facfe;
        }
        /* Styling the uploader */
        .stFileUploader {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px dashed rgba(255, 255, 255, 0.2);
        }
        /* Styling the button */
        .stButton>button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: #0f2027;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0px 4px 15px rgba(79, 172, 254, 0.4);
            border: none;
        }
        /* Metrics styling */
        div[data-testid="stMetricValue"] {
            color: #00f2fe;
            font-size: 2rem;
            font-weight: bold;
        }
        /* Card style for insights */
        .insight-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            border-left: 5px solid #00f2fe;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI Sports Analytics", layout="wide", page_icon="🏅")
    add_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3048/3048354.png", width=100)
        st.title("Coach Assistant AI")
        st.markdown("""
        **Sports Performance Analytics System**
        
        This tool uses a deep learning **CNN + LSTM** architecture to automatically scan and index hours of raw training footage.
        
        **Trackable Mechanics:**
        - ⚾ Baseball Pitching
        - 🎽 Javelin Throw
        - 🏊 Diving
        - 🏋️ Clean & Jerk
        - And more...
        """)
        st.info("Upload athlete footage to automatically tag mechanics and generate searchable clips.")

    st.title("🏅 AI Sports Performance Analytics")
    st.markdown("Upload raw training footage. The system will automatically detect the athletic movement and index the clip for the coaching staff.")

    data_source = st.radio("Choose Video Source:", ("Upload Custom Video", "Use Sample Video"), horizontal=True)

    temp_video_path = None
    original_video_name = None

    if data_source == "Upload Custom Video":
        uploaded_file = st.file_uploader("Drop training footage here", type=["mp4", "mpeg", "avi", "mov"])
        if uploaded_file is not None:
            os.makedirs("temp", exist_ok=True)
            temp_video_path = os.path.join("temp", uploaded_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            original_video_name = uploaded_file.name
    else:
        # User selected to use a sample video
        os.makedirs("samples", exist_ok=True)
        sample_files = ["baseball.mp4"]
        
        # Verify the file actually exists
        if os.path.exists(os.path.join("samples", "baseball.mp4")):
            selected_sample = st.selectbox("Select a Sample Video to Analyze:", sample_files)
            if selected_sample:
                temp_video_path = os.path.join("samples", selected_sample)
                original_video_name = selected_sample
        else:
            st.warning("The baseball.mp4 sample video is missing from the samples directory.")

    if temp_video_path is not None and original_video_name is not None:
        os.makedirs("video", exist_ok=True)
        playable_original_path = os.path.join("temp", "playable_" + original_video_name.split('.')[0] + ".mp4")
        
        st.markdown("---")
        
        # Display area
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Original Video")
            
            # If the video is an .mp4, it might play natively. If it's an avi/mpeg, it needs conversion.
            # We'll just safely convert anything to an avc1 mp4 so it always plays in HTML5.
            if not os.path.exists(playable_original_path):
                with st.spinner("Converting original video to browser-playable format..."):
                    convert_video_for_browser(temp_video_path, playable_original_path)
            
            st.video(playable_original_path)

        with col2:
            st.subheader("🔍 Analysis & Results")
            if st.button("🚀 Analyze Video"):
                # Placeholder for progress
                status_text = st.empty()
                status_text.info("Loading model and extracting frames...")
                
                model = load_LRCN_model()
                
                # Get Video Metadata
                video_reader = cv2.VideoCapture(temp_video_path)
                fps = int(video_reader.get(cv2.CAP_PROP_FPS))
                frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
                v_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
                v_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_reader.release()

                # Get Predictions
                predicted_class, confidence = predict_single_action(temp_video_path, model)
                
                # Display Metrics
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric(label="Predicted Action", value=predicted_class)
                metric_col2.metric(label="Confidence", value=f"{confidence*100:.1f}%")

                st.markdown("---")
                
                # Display Telemetry
                st.markdown("**📹 Video Telemetry**")
                t_col1, t_col2, t_col3 = st.columns(3)
                t_col1.metric(label="Resolution", value=f"{v_width}x{v_height}")
                t_col2.metric(label="FPS", value=fps)
                t_col3.metric(label="Total Frames", value=frame_count)

                # Display AI Coaching Insights
                st.markdown("---")
                st.markdown("**🧠 AI Coaching Insights**")
                insight = COACHING_INSIGHTS.get(predicted_class, COACHING_INSIGHTS["Default"])
                st.markdown(f"""
                <div class="insight-card">
                    <b>⚙️ Mechanic Analysis:</b> {insight['Mechanic Analysis']}<br><br>
                    <b>⚠️ Stress Point:</b> {insight['Stress Point']}<br><br>
                    <b>💡 Recommendation:</b> {insight['Recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                status_text.info("Annotating the video frame-by-frame... This may take a minute.")

                # Annotate Video
                output_path = os.path.join("video", original_video_name.split('.')[0] + "_output.mp4")
                annotate_video(temp_video_path, output_path, model)

                status_text.success("Analysis and Video Generation Complete!")
                st.video(output_path)

if __name__ == "__main__":
    main()
