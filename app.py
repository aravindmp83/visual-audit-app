import streamlit as st
import cv2
import numpy as np
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from datetime import datetime
import io

# --- 1. SETUP & AUTH ---
st.set_page_config(page_title="Smart Visual Audit v2", page_icon="🕵️")

# We use a single scope for everything
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 
          'https://www.googleapis.com/auth/drive']

def get_creds():
    creds_dict = dict(st.secrets["gcp_service_account"])
    return ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPES)

# --- 2. GOOGLE DRIVE UPLOAD FUNCTION ---
def upload_image_to_drive(file_obj, filename):
    """Uploads the photo to Drive and returns the Web Link."""
    creds = get_creds()
    service = build('drive', 'v3', credentials=creds)
    
    # search for the folder 'Audit_Evidence_Photos'
    query = "mimeType='application/vnd.google-apps.folder' and name='Audit_Evidence_Photos'"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    
    folder_id = None
    if not items:
        # Create if not exists
        file_metadata = {
            'name': 'Audit_Evidence_Photos',
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
    else:
        folder_id = items[0]['id']

    # Upload File
    file_metadata = {'name': filename, 'parents': [folder_id]}
    media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg')
    file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
    return file.get('webViewLink')

# --- 3. FETCH STORE LIST ---
@st.cache_data
def get_store_list():
    try:
        creds = get_creds()
        client = gspread.authorize(creds)
        sheet = client.open("Visual_Audit_Database").worksheet("Stores")
        return sheet.col_values(1)[1:] # Skip header
    except:
        return ["Error: Create 'Stores' tab in Sheet"]

# --- 4. IMPROVED AI ENGINE (KNN + LOWE'S RATIO TEST) ---
@st.cache_resource
def load_reference_signatures():
    signatures = {"current": [], "old": []}
    orb = cv2.ORB_create(nfeatures=2000) # Increased features
    base_dir = "references" 
    
    if not os.path.exists(base_dir):
        return signatures

    for category in ["current", "old"]:
        cat_path = os.path.join(base_dir, category)
        if os.path.exists(cat_path):
            for root, dirs, files in os.walk(cat_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        campaign_name = os.path.basename(root)
                        # If image is in the root category folder, use filename
                        if campaign_name == category: campaign_name = file
                        
                        full_path = os.path.join(root, file)
                        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            kp, des = orb.detectAndCompute(img, None)
                            if des is not None:
                                signatures[category].append({
                                    "campaign": campaign_name,
                                    "descriptors": des
                                })
    return signatures

def find_best_match(uploaded_image, signatures):
    # Convert upload to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img_input = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create(nfeatures=2000)
    kp_input, des_input = orb.detectAndCompute(img_input, None)
    
    if des_input is None: return None, "Blurry", 0, 0

    # KNN Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    best_match_score = 0
    best_campaign = "Unknown"
    detected_category = "None"
    
    # Helper to check a list of signatures
    def check_category(category_list):
        local_best = 0
        local_camp = ""
        for item in category_list:
            if item["descriptors"] is None or len(item["descriptors"]) < 2: continue
            
            # KNN Match (k=2)
            try:
                matches = bf.knnMatch(des_input, item["descriptors"], k=2)
                good = []
                # Lowe's Ratio Test (Strict filter)
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                
                score = len(good)
                if score > local_best:
                    local_best = score
                    local_camp = item["campaign"]
            except:
                continue
        return local_best, local_camp

    # Check Current
    score_current, name_current = check_category(signatures["current"])
    # Check Old
    score_old, name_old = check_category(signatures["old"])

    # LOGIC:
    # Threshold = 15 good matches (Adjust based on testing)
    THRESHOLD = 15 

    debug_info = f"Current Score: {score_current} ({name_current}) | Old Score: {score_old} ({name_old})"

    if score_current < THRESHOLD and score_old < THRESHOLD:
        return "None", "Unknown / No Match", 0, debug_info
    
    if score_current >= score_old:
        return "current", name_current, score_current, debug_info
    else:
        return "old", name_old, score_old, debug_info

# --- 5. THE UI ---
st.title("Visual Campaign Tracker v2")

# Load Resources
refs = load_reference_signatures()
store_list = get_store_list()

store_id = st.selectbox("Select Store Code", store_list)
photo = st.camera_input("Capture Store Visual")

if photo:
    with st.spinner("Analyzing & Uploading..."):
        # 1. AI Check
        category, campaign, score, debug_msg = find_best_match(photo, refs)
        
        # 2. Upload to Drive (Reset pointer first)
        photo.seek(0)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{store_id}_{timestamp_str}.jpg"
        image_link = upload_image_to_drive(photo, filename)
        
        # 3. Display Results
        status_msg = ""
        points = 0
        
        st.caption(f"Debug: {debug_msg}") # Helpful for you to see why it failed

        if category == "current":
            st.success(f"✅ Verified! Campaign: **{campaign}**")
            status_msg = "Compliant"
            points = 1
            st.balloons()
        elif category == "old":
            st.error(f"❌ Old Campaign Detected: **{campaign}**")
            status_msg = "Non-Compliant (Old)"
            points = 0
        else:
            st.warning("⚠️ No clear match found. Please retake closer.")
            status_msg = "Unknown / Retake"
            points = 0

        # 4. Save to Sheet
        try:
            creds = get_creds()
            client = gspread.authorize(creds)
            sheet = client.open("Visual_Audit_Database").sheet1 # 'Logs' tab
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet.append_row([timestamp, store_id, campaign, status_msg, points, score, image_link])
            st.toast("Saved to Database!")
        except Exception as e:
            st.error(f"Database Error: {e}")
