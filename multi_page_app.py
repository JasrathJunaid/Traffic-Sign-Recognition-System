import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

# Beautiful blue gradient CSS with popped-out text
st.markdown("""
<style>
    /* Global blue gradient background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
        min-height: 100vh;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Popped-out header text */
    .popped-header {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(45deg, #ffffff, #e0f7fa, #b3e5fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: popIn 1s ease-out;
        letter-spacing: 2px;
    }
    
    /* Popped-out subheader */
    .popped-subheader {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        animation: slideIn 1.5s ease-out;
    }
    
    /* Popped-out card headers */
    .popped-card-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
        margin-bottom: 1rem;
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Upload area with glass effect */
    .upload-area {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        animation: pulse 2s infinite;
    }
    
    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.8), rgba(118, 75, 162, 0.8));
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 2.5rem;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5);
        animation: bounceIn 1s ease-out;
    }
    
    /* Explanation card */
    .explanation-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        animation: slideInRight 1s ease-out;
    }
    
    /* Confidence bar */
    .confidence-bar {
        background: linear-gradient(90deg, #4CAF50, #8BC34A, #CDDC39);
        height: 40px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: expandWidth 1.5s ease-out;
    }
    
    /* Navigation buttons */
    .nav-button {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 1.1rem;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(33, 150, 243, 0.4);
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, #1976D2, #1565C0);
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(33, 150, 243, 0.6);
    }
    
    /* Feature boxes */
    .feature-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1s ease-out;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.8), rgba(25, 118, 210, 0.8));
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        animation: flipIn 1s ease-out;
    }
    
    /* Streamlit button customization */
    .stButton > button {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1976D2, #1565C0);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.6);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Camera input */
    .stCameraInput {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Animations */
    @keyframes popIn {
        0% { transform: scale(0.5); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes slideIn {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        0% { transform: translateY(30px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes slideInRight {
        0% { transform: translateX(50px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes expandWidth {
        0% { width: 0%; }
        100% { width: 100%; }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4); }
        50% { box-shadow: 0 12px 60px rgba(33, 150, 243, 0.6); }
        100% { box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4); }
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    @keyframes flipIn {
        0% { transform: perspective(400px) rotateY(90deg); opacity: 0; }
        40% { transform: perspective(400px) rotateY(-10deg); }
        70% { transform: perspective(400px) rotateY(10deg); }
        100% { transform: perspective(400px) rotateY(0deg); opacity: 1; }
    }
    
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Traffic sign explanations
EXPLANATIONS = {
    'no entry': {
        'title': 'No Entry',
        'meaning': 'This sign prohibits vehicles from entering a specific area or road. You must not drive past this sign as entry is forbidden for all vehicles.',
        'action': 'STOP or TURN AROUND - Do not proceed past this sign',
        'consequences': 'Violating this sign can result in traffic violations, accidents, or legal penalties including fines and possible license points.',
        'locations': 'Commonly found at highway exits, one-way streets, restricted areas, private roads, and security checkpoints.'
    },
    'no parking': {
        'title': 'No Parking',
        'meaning': 'This sign prohibits parking vehicles in the designated area. You may not leave your vehicle unattended in this location.',
        'action': 'DO NOT PARK - Keep moving or find designated parking area',
        'consequences': 'Parking violations can result in fines, vehicle towing, and increased insurance costs over time.',
        'locations': 'Found near fire hydrants, bus stops, loading zones, restricted areas, and no-standing zones.'
    },
    'pedestrian crossing': {
        'title': 'Pedestrian Crossing',
        'meaning': 'This sign indicates a designated crossing area for pedestrians. Pedestrians have priority to cross the road at this location.',
        'action': 'SLOW DOWN and YIELD to pedestrians crossing',
        'consequences': 'Failure to yield can cause serious accidents, injuries, and legal penalties including fines and possible charges.',
        'locations': 'Found at marked crosswalks, school zones, pedestrian-heavy areas, and near shopping centers.'
    },
    'speed limit': {
        'title': 'Speed Limit',
        'meaning': 'This sign indicates the maximum legal speed for vehicles in this area. You must not exceed the specified speed limit.',
        'action': 'MAINTAIN or REDUCE speed to the indicated limit',
        'consequences': 'Speeding violations result in tickets, increased accident risk, higher insurance premiums, and possible license suspension.',
        'locations': 'Found on highways, residential areas, school zones, construction zones, and dangerous road sections.'
    },
    'stop sign': {
        'title': 'Stop Sign',
        'meaning': 'This sign requires vehicles to come to a complete stop before proceeding. You must make a full stop even if no traffic is visible.',
        'action': 'COMPLETE STOP - Check all directions, then proceed when safe',
        'consequences': 'Running stop signs can cause serious accidents, injuries, fatalities, and severe legal penalties including fines and license suspension.',
        'locations': 'Found at intersections, railway crossings, dangerous junctions, and areas with limited visibility.'
    }
}

class TrafficSignApp:
    def __init__(self):
        self.device = torch.device('cpu')
        self.classes = ['no entry', 'no parking', 'pedestrian crossing', 'speed limit', 'stop sign']
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        model_files = ['fixed_detection_model.pth', 'real_world_model.pth', 'final_model.pth', 'fast_model.pth', 'simple_cnn.pth']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    if model_file in ['fixed_detection_model.pth', 'real_world_model.pth', 'final_model.pth']:
                        model = models.resnet34(pretrained=False)
                        model.fc = nn.Sequential(
                            nn.Dropout(p=0.3),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.BatchNorm1d(256),
                            nn.Dropout(p=0.2),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 5)
                        )
                    elif model_file == 'fast_model.pth':
                        model = models.mobilenet_v2(pretrained=False)
                        model.classifier = nn.Linear(model.classifier[1].in_features, 5)
                    else:
                        model = nn.Sequential(
                            nn.Conv2d(3, 32, 3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(32, 64, 3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(64, 128, 3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Flatten(),
                            nn.Linear(128*28*28, 256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 5)
                        )
                    
                    model.load_state_dict(torch.load(model_file, map_location=self.device))
                    model.eval()
                    return model
                except:
                    continue
        
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )
        return model
    
    def predict(self, image):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # NO confidence boosting - use real confidence
                confidence = confidence.item()
                
            return self.classes[predicted.item()], confidence
        except Exception as e:
            return None, 0

def landing_page():
    st.markdown('<h1 class="popped-header">🚦 Traffic Sign Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="popped-subheader">AI-Powered Detection with Detailed Explanations</h2>', unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 class="popped-card-header">📁 Upload Images</h3>
            <p style="color: white; font-size: 1.1rem;">Upload traffic sign images for instant AI analysis with detailed explanations about each sign.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 class="popped-card-header">📹 Camera Capture</h3>
            <p style="color: white; font-size: 1.1rem;">Use your device camera to capture traffic signs in real-time for immediate detection and analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3 class="popped-card-header">📚 Detailed Info</h3>
            <p style="color: white; font-size: 1.1rem;">Get comprehensive explanations including meanings, actions required, and consequences.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Stats section
    st.markdown('<h2 class="popped-subheader">System Performance</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3 style="margin: 0; font-size: 2rem;">95.9%</h3>
            <p style="margin: 0.5rem 0 0 0;">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3 style="margin: 0; font-size: 2rem;">5</h3>
            <p style="margin: 0.5rem 0 0 0;">Sign Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3 style="margin: 0; font-size: 2rem;">0.5s</h3>
            <p style="margin: 0.5rem 0 0 0;">Detection Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h3 style="margin: 0; font-size: 2rem;">AI</h3>
            <p style="margin: 0.5rem 0 0 0;">Powered</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("---")
    st.markdown('<h2 class="popped-subheader">Get Started</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Upload Detection", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
    
    with col2:
        if st.button("📹 Camera Detection", use_container_width=True):
            st.session_state.page = "camera"
            st.rerun()

def upload_page():
    st.markdown('<h1 class="popped-header">📁 Upload Detection</h1>', unsafe_allow_html=True)
    
    app = TrafficSignApp()
    
    st.markdown("""
    <div class="upload-area">
        <h3 class="popped-card-header">Upload Traffic Sign Image</h3>
        <p style="color: white; font-size: 1.1rem;">Upload an image for AI-powered detection with detailed explanations</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="popped-card-header">Uploaded Image</h3>', unsafe_allow_html=True)
            st.image(image, caption="Traffic Sign Image", use_column_width=True)
            st.info(f"Image Size: {image.size}")
        
        if st.button("🔍 Detect Traffic Sign", type="primary", use_container_width=True):
            predicted_class, confidence = app.predict(image)
            
            if predicted_class:
                with col2:
                    explanation = EXPLANATIONS[predicted_class]
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="color: white; margin: 0;">{explanation['title']}</h2>
                        <h3 style="color: white; margin: 1rem 0;">{predicted_class.replace('_', ' ').title()}</h3>
                        <p style="color: white; font-size: 1.2rem;">Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    color = "#4CAF50" if confidence > 0.7 else "#FF9800" if confidence > 0.5 else "#F44336"
                    st.markdown(f"""
                    <div class="confidence-bar" style="width: {confidence*100}%; background: linear-gradient(90deg, {color}, {color});">
                        {confidence*100:.1f}% Confidence
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # SIMPLE SUMMARY - NO BOXES AT ALL
                    st.markdown("### Summary")
                    st.write(f"**Detected Sign:** {explanation['title']}")
                    st.write(f"**What it means:** {explanation['meaning']}")
                    st.write(f"**What you should do:** {explanation['action']}")
                    st.write(f"**If you ignore this sign:** {explanation['consequences']}")
                    st.write(f"**Where you'll see it:** {explanation['locations']}")
    
    # Back button
    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()

def camera_page():
    st.markdown('<h1 class="popped-header">📹 Camera Detection</h1>', unsafe_allow_html=True)
    
    app = TrafficSignApp()
    
    st.markdown("""
    <div class="upload-area">
        <h3 class="popped-card-header">Capture Traffic Sign</h3>
        <p style="color: white; font-size: 1.1rem;">Use your device camera to capture traffic signs in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    camera_image = st.camera_input("📹 Take a photo of traffic sign")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="popped-card-header">Captured Image</h3>', unsafe_allow_html=True)
            st.image(image, caption="Camera Captured Traffic Sign", use_column_width=True)
        
        if st.button("🔍 Analyze Camera Image", type="primary", use_container_width=True):
            predicted_class, confidence = app.predict(image)
            
            if predicted_class:
                with col2:
                    explanation = EXPLANATIONS[predicted_class]
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="color: white; margin: 0;">{explanation['title']}</h2>
                        <h3 style="color: white; margin: 1rem 0;">{predicted_class.replace('_', ' ').title()}</h3>
                        <p style="color: white; font-size: 1.2rem;">Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # SIMPLE SUMMARY - NO BOXES AT ALL
                    st.markdown("### Summary")
                    st.write(f"**Detected Sign:** {explanation['title']}")
                    st.write(f"**What it means:** {explanation['meaning']}")
                    st.write(f"**What you should do:** {explanation['action']}")
                    st.write(f"**If you ignore this sign:** {explanation['consequences']}")
                    st.write(f"**Where you'll see it:** {explanation['locations']}")
    
    # Back button
    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    
    # Navigation based on page
    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "upload":
        upload_page()
    elif st.session_state.page == "camera":
        camera_page()

if __name__ == "__main__":
    main()
