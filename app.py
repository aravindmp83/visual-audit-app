import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import gspread

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Store Visual Repository", page_icon="📂")

# 🔴 IMPORTANT: PASTE YOUR FOLDER ID INSIDE THE QUOTES BELOW 🔴
# Example: TARGET_FOLDER_ID = "1HaBcD_eFgHiJkLmNoPqRsTuVwXyZ"
TARGET_FOLDER_ID = "1u5pllOyCTfKQEJk6y_Q4PZNI4nr0xuWi" 

SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']

# Marketing Elements List
ELEMENT_TYPES = [
    "Totem Pole",
    "Flag Pole",
    "Hoarding",
    "Facade",
    "Window Display",
    "Cash Counter",
    "Entrance Arch",
    "Store Signage"
]

# Status / Damage Options
STATUS_OPTIONS = [
    "Intact and Working",
    "Flex Damage",
    "Frame Damage",
    "Total Damage",
    "Letter Damage"
]

# --- 2. GOOGLE SERVICES SETUP ---
def get_creds():
    """Authenticates using the secrets file."""
    creds_dict = dict(st.secrets["gcp_service_account"])
    return ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPES)

def get_store_list():
    """Fetches store list from Google Sheets."""
    try:
        creds = get_creds()
        client = gspread.authorize(creds)
        # Ensure you have a tab named 'Stores' in your sheet
        sheet = client.open("Visual_Audit_Database").worksheet("Stores")
        return sheet.col_values(1)[1:] 
    except Exception as e:
        return [f"Connection Error: {e}"]

def upload_to_drive(file_obj, filename):
    """Uploads the file directly to the Target Folder ID."""
    creds = get_creds()
    service = build('drive', 'v3', credentials=creds)
    
    file_metadata = {
        'name': filename,
        'parents': [TARGET_FOLDER_ID] # This forces it into YOUR drive
    }
    media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg')
    
    file = service.files().create(
        body=file_metadata, 
        media_body=media, 
        fields='id, webViewLink'
    ).execute()
    
    return file.get('webViewLink')

def log_to_sheet(timestamp, store, element, status, link):
    """Logs the entry to the Google Sheet."""
    try:
        creds = get_creds()
        client = gspread.authorize(creds)
        sheet = client.open("Visual_Audit_Database").sheet1
        # Added 'status' to the log row
        sheet.append_row([timestamp, store, element, status, "Uploaded", link])
    except Exception as e:
        st.error(f"Could not log to sheet: {e}")

# --- 3. THE APP INTERFACE ---
st.title("Store Visual Repository 📸")
st.write("Upload latest visual evidence.")

# Inputs
store_list = get_store_list()
selected_store = st.selectbox("Select Store", store_list)
selected_element = st.selectbox("Select Visual Element", ELEMENT_TYPES)
selected_status = st.selectbox("Condition Status", STATUS_OPTIONS)

# Camera
photo = st.camera_input(f"Take photo of {selected_element}")

if photo:
    with st.spinner("Saving to Repository..."):
        try:
            # 1. Construct Filename
            # Format: Store_Element_Status_Date.jpg
            clean_element = selected_element.replace(" ", "")
            clean_status = selected_status.replace(" ", "")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"{selected_store}_{clean_element}_{clean_status}_{timestamp_str}.jpg"
            
            # 2. Upload
            photo.seek(0)
            link = upload_to_drive(photo, filename)
            
            # 3. Log
            log_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_to_sheet(log_timestamp, selected_store, selected_element, selected_status, link)
            
            st.success(f"✅ Saved: {selected_element} ({selected_status})")
            
        except Exception as e:
            st.error(f"Upload Error: {e}")
            st.info("Check: Did you paste the correct Folder ID in line 13?")
