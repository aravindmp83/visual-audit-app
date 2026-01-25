import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import gspread

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Visual Data Collector", page_icon="📁")

# Define your specific scopes
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']

# Marketing Elements List
ELEMENT_TYPES = [
    "Totem Pole",
    "Flag Pole",
    "Hoarding",
    "Facade",
    "Window Display",
    "Cash Counter",
    "Entrance Arch"
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
        return sheet.col_values(1)[1:] # Returns list of stores, skipping header
    except Exception as e:
        return [f"Error loading stores: {e}"]

def upload_to_drive(file_obj, filename, store_name):
    """Uploads the file to a specific folder in Drive."""
    creds = get_creds()
    service = build('drive', 'v3', credentials=creds)
    
    # 1. Find or Create a folder named "Store_Visuals_Data"
    folder_name = "Store_Visuals_Data"
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    
    if not items:
        # Create the folder if it doesn't exist
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
    else:
        folder_id = items[0]['id']

    # 2. Upload the File
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg')
    
    file = service.files().create(
        body=file_metadata, 
        media_body=media, 
        fields='id, webViewLink'
    ).execute()
    
    return file.get('webViewLink')

def log_to_sheet(timestamp, store, element, link):
    """Logs the entry to the Google Sheet for tracking."""
    try:
        creds = get_creds()
        client = gspread.authorize(creds)
        sheet = client.open("Visual_Audit_Database").sheet1
        sheet.append_row([timestamp, store, element, "Uploaded", link])
    except Exception as e:
        st.error(f"Could not log to sheet: {e}")

# --- 3. THE APP INTERFACE ---
st.title("Store Visual Collector 📸")
st.info("Capture store elements for Central Audit.")

# Inputs
store_list = get_store_list()
selected_store = st.selectbox("Select Store", store_list)
selected_element = st.selectbox("Select Marketing Element", ELEMENT_TYPES)

# Camera
photo = st.camera_input(f"Take photo of {selected_element}")

if photo:
    # Submit Button logic is handled by the presence of the photo
    with st.spinner("Uploading to HQ Server..."):
        try:
            # 1. Construct Filename
            # Format: StoreName_ElementType_Date_Time.jpg
            clean_element = selected_element.replace(" ", "")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{selected_store}_{clean_element}_{timestamp_str}.jpg"
            
            # 2. Upload
            photo.seek(0)
            link = upload_to_drive(photo, filename, selected_store)
            
            # 3. Log
            log_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_to_sheet(log_timestamp, selected_store, selected_element, link)
            
            st.success(f"✅ Uploaded: {filename}")
            st.toast("Success! Ready for next photo.")
            
        except Exception as e:
            st.error(f"Upload Failed: {e}")
            st.warning("Please ensure the 'Visual_Audit_Bot' email has Editor access to your Google Drive.")
