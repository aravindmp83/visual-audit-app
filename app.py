import streamlit as st
import cv2
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- CONFIGURATION ---
# Define the visuals you want to check
VISUAL_SLOTS = ["Main Window", "Cash Counter"]
STORES = [f"Store_{i}" for i in range(1, 201)]

# --- SETUP GOOGLE SHEETS CONNECTION ---
def get_google_sheet():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 
             'https://www.googleapis.com/auth/drive']
    # Load secrets from Streamlit Cloud
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open("Visual_Audit_Database").sheet1

# --- IMAGE MATCHING LOGIC ---
def analyze_image(uploaded_image, reference_path):
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img_input = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Load Reference Image
    img_ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    
    if img_ref is None:
        return 0 # Error loading reference
        
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img_input, None)
    kp2, des2 = orb.detectAndCompute(img_ref, None)
    
    # Match Features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate Score (Top 15% of matches)
    good_matches = [m for m in matches if m.distance < 50]
    score = len(good_matches)
    return score

# --- THE APP INTERFACE ---
st.title("Visual Compliance Audit 📸")

store_id = st.selectbox("Select Store", STORES)
visual_type = st.selectbox("Select Visual", VISUAL_SLOTS)
photo = st.camera_input("Take a photo of the visual")

if photo:
    with st.spinner("Analyzing Compliance..."):
        # 1. Check against CURRENT Campaign
        # Note: Ensure 'references/current_window.jpg' exists in your repo
        score_current = analyze_image(photo, "references/current_window.jpg")
        
        # 2. Check against OLD Campaign
        photo.seek(0) # Reset file pointer
        score_old = analyze_image(photo, "references/old_window.jpg")
        
        final_status = "Unknown"
        points = 0
        
        # Threshold Logic (Adjust '20' based on testing)
        if score_current > 20:
            final_status = "✅ Compliant (Current)"
            points = 1
            st.success(f"Verified! Found Current Campaign. (Score: {score_current})")
        elif score_old > 20:
            final_status = "⚠️ Non-Compliant (Old Visual Found)"
            points = 0
            st.error(f"Alert! Old Campaign Detected. (Score: {score_old})")
        else:
            final_status = "❌ Missing / Unclear"
            points = 0
            st.warning("Could not identify visual. Please retake closer.")

        # 3. Save to Google Sheet
        try:
            sheet = get_google_sheet()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet.append_row([timestamp, store_id, visual_type, final_status, points])
            st.toast("Audit Saved to Database!")
        except Exception as e:
            st.error(f"Database Error: {e}")
