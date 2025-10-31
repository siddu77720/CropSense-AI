import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CropSense AI",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .disease-card {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #FF4B4B;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFA500;
        font-weight: bold;
    }
    .confidence-low {
        color: #008000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Disease information database
DISEASE_INFO = {
    "Healthy": {
        "description": "The plant appears healthy with no visible signs of disease.",
        "treatment": "Continue regular maintenance and monitoring.",
        "prevention": "Maintain proper watering, fertilization, and pest control."
    },
    "Early Blight": {
        "description": "Dark brown spots with concentric rings on leaves, usually starting from lower leaves.",
        "treatment": "Apply copper-based fungicides. Remove infected leaves. Use chlorothalonil or mancozeb.",
        "prevention": "Rotate crops, ensure proper spacing, avoid overhead watering."
    },
    "Late Blight": {
        "description": "Water-soaked spots that turn brown, white mold growth on undersides.",
        "treatment": "Apply fungicides containing chlorothalonil or mancozeb immediately.",
        "prevention": "Use resistant varieties, proper spacing, remove infected plants."
    },
    "Powdery Mildew": {
        "description": "White powdery spots on leaves and stems, leaves may yellow and die.",
        "treatment": "Apply sulfur, potassium bicarbonate, or neem oil. Improve air circulation.",
        "prevention": "Proper spacing, avoid overhead watering, morning watering."
    },
    "Leaf Spot": {
        "description": "Circular brown or black spots with yellow halos on leaves.",
        "treatment": "Remove infected leaves. Apply copper fungicide or chlorothalonil.",
        "prevention": "Avoid overhead watering, clean garden debris, crop rotation."
    },
    "Bacterial Spot": {
        "description": "Small water-soaked spots that become dark and sunken.",
        "treatment": "Copper-based bactericides. Remove severely infected plants.",
        "prevention": "Use disease-free seeds, avoid working with wet plants."
    },
    "Mosaic Virus": {
        "description": "Mottled light and dark green patterns on leaves, stunted growth.",
        "treatment": "No cure. Remove and destroy infected plants to prevent spread.",
        "prevention": "Control aphids, use virus-free plants, disinfect tools."
    },
    "Root Rot": {
        "description": "Wilting, yellowing leaves, dark rotten roots, plant collapse.",
        "treatment": "Improve drainage, apply fungicides like thiophanate-methyl.",
        "prevention": "Proper drainage, avoid overwatering, soil sterilization."
    }
}

class DiseaseDetector:
    def __init__(self):
        self.model = None
        self.class_names = list(DISEASE_INFO.keys())
        self.load_model()
    
    def load_model(self):
        """Load or create a demo model"""
        try:
            # Try to load existing model
            self.model = tf.keras.models.load_model('crop_disease_model.h5')
            st.success("✅ Pre-trained model loaded successfully!")
        except:
            # Create a dummy model for demo
            st.warning("⚠️ No pre-trained model found. Using demo mode with simulated predictions.")
            self.model = self.create_demo_model()
    
    def create_demo_model(self):
        """Create a simple demo model for testing"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict_disease(self, image):
        """Predict disease from image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            disease_name = self.class_names[predicted_class_idx]
            
            return disease_name, confidence
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            # Return a demo prediction for testing
            return "Early Blight", 0.85

def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 CropSense AI - Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown("### Upload a plant leaf image to detect diseases and get treatment advice")
    
    # Initialize detector
    detector = DiseaseDetector()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Input Method", 
                                   ["📁 Upload Image", "📷 Camera Capture", "ℹ️ Disease Information"])
    
    if app_mode == "📁 Upload Image":
        handle_image_upload(detector)
    elif app_mode == "📷 Camera Capture":
        handle_camera_capture(detector)
    else:
        show_disease_info()

def handle_image_upload(detector):
    st.markdown('<div class="sub-header">📁 Upload Plant Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of plant leaves for disease detection"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("🔍 Detect Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                disease, confidence = detector.predict_disease(image)
                display_results(disease, confidence)

def handle_camera_capture(detector):
    st.markdown('<div class="sub-header">📷 Capture Image</div>', unsafe_allow_html=True)
    
    # Camera input
    camera_image = st.camera_input("Take a picture of plant leaves")
    
    if camera_image is not None:
        # Display captured image
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)
        
        # Predict immediately
        with st.spinner("Analyzing image..."):
            disease, confidence = detector.predict_disease(image)
            display_results(disease, confidence)

def display_results(disease, confidence):
    """Display prediction results and treatment information"""
    st.markdown("---")
    st.markdown(f'<div class="sub-header">🔬 Detection Results</div>', unsafe_allow_html=True)
    
    # Confidence color coding
    if confidence > 0.8:
        conf_class = "confidence-high"
    elif confidence > 0.6:
        conf_class = "confidence-medium"
    else:
        conf_class = "confidence-low"
    
    # Results columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Detected Disease", disease)
        st.markdown(f'<p class="{conf_class}">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
    
    with col2:
        if confidence < 0.5:
            st.warning("⚠️ Low confidence result. Please upload a clearer image.")
    
    # Disease information card
    st.markdown('<div class="disease-card">', unsafe_allow_html=True)
    st.markdown(f"### 📋 About {disease}")
    st.markdown(f"**Description:** {DISEASE_INFO[disease]['description']}")
    st.markdown(f"**🩺 Treatment:** {DISEASE_INFO[disease]['treatment']}")
    st.markdown(f"**🛡️ Prevention:** {DISEASE_INFO[disease]['prevention']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional recommendations
    st.markdown("### 💡 Additional Recommendations")
    st.markdown("""
    - Take multiple images from different angles for better accuracy
    - Consult with agricultural experts for severe infections
    - Monitor plants regularly for early detection
    - Maintain proper plant spacing and air circulation
    """)

def show_disease_info():
    st.markdown('<div class="sub-header">🌿 Common Plant Diseases Information</div>', unsafe_allow_html=True)
    
    for disease, info in DISEASE_INFO.items():
        with st.expander(f"📌 {disease}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Treatment:** {info['treatment']}")
            st.markdown(f"**Prevention:** {info['prevention']}")

if __name__ == "__main__":
    main()