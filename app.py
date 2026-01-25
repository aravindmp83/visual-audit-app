import streamlit as st
import cv2
import numpy as np
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- 1. SETUP & AUTH ---
st.set_page_config(page_title="Smart Visual Audit", page_icon="🕵️")

# Google Sheets Auth
def get_google_sheet():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 
             'https://www.googleapis.com/auth/drive']
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    # Ensure your Google Sheet is named exactly this:
    return client.open("Visual_Audit_Database").sheet1

# --- 2. THE AI ENGINE (Multi-Folder Support) ---

@st.cache_resource
def load_reference_signatures():
    """
    Scans the 'references' folder and memorizes the features of every campaign image.
    Returns a dictionary of signatures.
    """
    signatures = {"current": [], "old": []}
    orb = cv2.ORB_create()
    
    # We look for a folder named 'references' in the same directory as this script
    base_dir = "references" 
    
    if not os.path.exists(base_dir):
        st.error(f"⚠️ Error: Could not find '{base_dir}' folder in GitHub repo.")
        return signatures

    # Walk through folders
    for category in ["current", "old"]:
        cat_path = os.path.join(base_dir, category)
        if os.path.exists(cat_path):
            for root, dirs, files in os.walk(cat_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Get Campaign Name from the sub-folder name
                        campaign_name = os.path.basename(root) 
                        full_path = os.path.join(root, file)
                        
                        # Read and process image
                        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            kp, des = orb.detectAndCompute(img, None)
                            if des is not None:
                                signatures[category].append({
                                    "campaign": campaign_name,
                                    "filename": file,
                                    "descriptors": des
                                })
    
    return signatures

def find_best_match(uploaded_image, signatures):
    """
    Compares upload against ALL loaded signatures.
    """
    # Convert upload to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img_input = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create()
    kp_input, des_input = orb.detectAndCompute(img_input, None)
    
    if des_input is None:
        return None, "Image too blurry", 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    best_match_score = 0
    best_campaign = "Unknown"
    detected_category = "None"

    # Check against CURRENT first
    for item in signatures["current"]:
        matches = bf.match(des_input, item["descriptors"])
        # Score = number of similar features found
        score = len(matches) 
        if score > best_match_score:
            best_match_score = score
            best_campaign = item["campaign"]
            detected_category = "current"

    # If no strong match in current, check OLD
    # Threshold: 30 matches is a decent baseline for "Simlarity"
    if best_match_score < 30: 
        for item in signatures["old"]:
            matches = bf.match(des_input, item["descriptors"])
            score = len(matches)
            if score > best_match_score:
                best_match_score = score
                best_campaign = item["campaign"]
                detected_category = "old"

    return detected_category, best_campaign, best_match_score

# --- 3. THE USER INTERFACE ---

st.title("Visual Campaign Tracker 📸")

# Load the AI Brain (Happens once)
refs = load_reference_signatures()
st.write(f"System Loaded: {len(refs['current'])} Current & {len(refs['old'])} Old Visuals.")

store_id = st.selectbox("Select Store", [f"Store_{i}" for i in range(1, 201)])
photo = st.camera_input("Capture Store Visual")

if photo:
    with st.spinner("AI is scanning all campaigns..."):
        category, campaign, score = find_best_match(photo, refs)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_msg = ""
        points = 0
        
        # LOGIC FOR SCORING
        if score < 30: # No match found
            st.warning("⚠️ No matching campaign found. Is the photo clear?")
            status_msg = "Unknown / No Match"
            points = 0
            
        elif category == "current":
            st.success(f"✅ Verified! Campaign: **{campaign}**")
            st.caption(f"Confidence Score: {score}")
            status_msg = f"Compliant: {campaign}"
            points = 1
            st.balloons()
            
        elif category == "old":
            st.error(f"❌ Alert! Old Campaign Detected: **{campaign}**")
            st.caption("Please remove this visual immediately.")
            status_msg = f"Non-Compliant: {campaign} (Old)"
            points = 0

        # SAVE TO SHEET
        if score >= 30: # Only save if we actually found something
            try:
                sheet = get_google_sheet()
                sheet.append_row([timestamp, store_id, campaign, status_msg, points])
                st.toast(f"Data saved for {store_id}")
            except Exception as e:
                st.error(f"Database Error: {e}")
