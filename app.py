import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import gspread

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Store Visual Repository", page_icon="📂")

# PASTE YOUR FOLDER ID HERE (From Step 1)
# Example: TARGET_FOLDER_ID = "1aBcD_eFgHiJkLmNoPqRsTuVwXyZ"
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
        'parents': [TARGET_FOLDER_ID] # This forces it into YOUR drive, not the Robot's
    }
    media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg')
    
    file = service.files().create(
        body=file_metadata, 
        media_body=media, 
        fields='id, webViewLink'
    ).execute()
    
    return file.get('webViewLink')

def log_to_sheet(timestamp, store, element, link):
    """Logs the entry to the Google Sheet."""
    try:
        creds = get_creds()
        client = gspread.authorize(creds)
        sheet = client.open("Visual_Audit_Database").sheet1
        sheet.append_row([timestamp, store, element, "Uploaded", link])
    except Exception as e:
        st.error(f"Could not log to sheet: {e}")

# --- 3. THE APP INTERFACE ---
st.title("Store Visual Collector 📸")
st.write("Please upload the latest visual for your store.")

# Inputs
store_list = get_store_list()
selected_store = st.selectbox("Select Store", store_list)
selected_element = st.selectbox("Select Visual Element", ELEMENT_TYPES)

# Camera
photo = st.camera_input(f"Take photo of {selected_element}")

if photo:
    with st.spinner("Uploading Image..."):
        try:
            # 1. Construct Filename
            clean_element = selected_element.replace(" ", "")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{selected_store}_{clean_element}_{timestamp_str}.jpg"
            
            # 2. Upload
            photo.seek(0)
            link = upload_to_drive(photo, filename)
            
            # 3. Log
            log_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_to_sheet(log_timestamp, selected_store, selected_element, link)
            
            st.success(f"✅ Uploaded successfully")
            
        except Exception as e:
            st.error(f"Upload Error: {e}")
            st.info("Check: Did you paste the correct Folder ID in the code?")
