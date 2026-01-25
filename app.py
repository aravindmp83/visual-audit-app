import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import pandas as pd

# --- 1. SETUP ---
st.set_page_config(page_title="Visual Collector (Supabase)", page_icon="⚡")

# Initialize connection
@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = init_connection()

# Constants
ELEMENT_TYPES = ["Totem Pole", "Flag Pole", "Hoarding", "Facade", "Window Display", "Cash Counter", "Store Signage"]
STATUS_OPTIONS = ["Intact and Working", "Flex Damage", "Frame Damage", "Total Damage", "Letter Damage"]

# --- 2. FUNCTIONS ---

def get_store_list():
    """Fetches the list of stores from the 'stores' table."""
    try:
        response = supabase.table("stores").select("store_code").execute()
        # Extract just the codes into a list
        return [row['store_code'] for row in response.data]
    except Exception as e:
        return [f"Error fetching stores: {e}"]

def upload_image(file_obj, filename):
    """Uploads image to Supabase Storage and returns the public URL."""
    try:
        bucket_name = "evidence-photos"
        file_bytes = file_obj.getvalue()
        
        # Upload
        supabase.storage.from_(bucket_name).upload(
            path=filename,
            file=file_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        # Get Public URL
        project_url = st.secrets["supabase"]["url"]
        public_url = f"{project_url}/storage/v1/object/public/{bucket_name}/{filename}"
        return public_url
        
    except Exception as e:
        st.error(f"Upload Error: {e}")
        return None

def save_log(store, element, status, image_url):
    """Inserts a new row into the 'audit_logs' table."""
    try:
        data = {
            "store_code": store,
            "element_type": element,
            "condition_status": status,
            "image_url": image_url
        }
        supabase.table("audit_logs").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

# --- 3. APP INTERFACE ---
st.title("Store Visual Collector ⚡")
st.caption("Powered by Supabase")

# Fetch Stores
store_options = get_store_list()
if not store_options:
    st.error("Could not load store list. Check database connection.")

# Form
selected_store = st.selectbox("Select Store", store_options)
selected_element = st.selectbox("Visual Element", ELEMENT_TYPES)
selected_status = st.selectbox("Condition", STATUS_OPTIONS)

photo = st.camera_input(f"Take photo of {selected_element}")

if photo:
    with st.spinner("Syncing to Database..."):
        # 1. Generate Filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_element = selected_element.replace(" ", "")
        filename = f"{selected_store}/{clean_element}_{timestamp}.jpg"
        
        # 2. Upload Image
        image_url = upload_image(photo, filename)
        
        if image_url:
            # 3. Save Record
            success = save_log(selected_store, selected_element, selected_status, image_url)
            
            if success:
                st.success("✅ Saved Successfully!")
                st.toast("Evidence Logged.")
                
                # Optional: Show preview of what was saved
                st.write(f"**Log ID:** {filename}")
                st.write(f"**Status:** {selected_status}")
