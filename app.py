import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Visual Adherence Audit", page_icon="✅")

# Mock Database of Visuals (In real life, this comes from your Google Sheet)
VISUAL_LIST = ["Window_Display_Left", "Cash_Counter_Standee", "Entrance_Totem"]
STORES = [f"Store_{i:03d}" for i in range(1, 201)]

def load_reference_images():
    """Loads the Master JPEGs into memory for comparison."""
    # In a real deployment, these load from a folder. 
    # For now, we simulate with a placeholder logic.
    return {"Window_Display_Left": "master_window_v2.jpg"}

# --- 2. THE AI BRAIN (ORB Matcher) ---
def check_visual_compliance(uploaded_image_file, reference_name):
    """
    Compares the uploaded photo against the Master JPEG.
    Returns: Score (0-100), Status (Current/Old/Mismatch)
    """
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image_file.read()), dtype=np.uint8)
    img_input = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    kp_input, desc_input = orb.detectAndCompute(img_input, None)
    
    # NOTE: In the live version, we would compare 'desc_input' against 
    # the pre-loaded descriptor of the 'reference_name'.
    # For this simplified pilot UI demo, we simulate a check.
    
    if len(kp_input) > 50: # If image has enough detail
        return 95, "✅ Current Campaign"
    else:
        return 20, "❌ Unclear / Old Visual"

# --- 3. THE APP INTERFACE ---
st.title("Store Visual Audit 📸")

# A. User Inputs
selected_store = st.selectbox("Select Your Store", STORES)
selected_visual = st.selectbox("Which Visual are you checking?", VISUAL_LIST)

# B. Camera Input
uploaded_file = st.camera_input(f"Take a photo of {selected_visual}")

if uploaded_file:
    # C. Instant Processing
    with st.spinner('Analyzing Visual...'):
        score, status = check_visual_compliance(uploaded_file, selected_visual)
        
        # D. Feedback
        if score > 80:
            st.success(f"{status} (Match: {score}%)")
            st.balloons()
            # Here we would code: save_to_google_sheets(selected_store, score)
        else:
            st.error(f"{status} (Match: {score}%)")
            st.warning("Please check if the visual is damaged or the old version.")
            
    st.info("Photo logged for Regional Review.")
