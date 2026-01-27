import streamlit as st
from supabase import create_client, Client
from datetime import datetime, date
import cv2
import numpy as np
import requests
import time
import pandas as pd

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Retail Visual Audit", page_icon="🏢", layout="wide")

@st.cache_resource
def init_connection():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except:
        return None

supabase: Client = init_connection()

ELEMENT_TYPES = ["Totem Pole", "Backlit Board", "Flagpole/Lollypop", "Façade", "Select your signage", "Others"]
STATUS_OPTIONS = ["Intact", "Flex Damage", "Frame Damage", "Total Damage", "Letter Damage"]

# --- 2. AI ENGINE (FIXED: INDENTATION & LOGIC) ---
@st.cache_resource
def load_reference_memory():
    # Fetch active campaigns (Start <= Today <= End)
    today = date.today().isoformat()
    active_camps = supabase.table("campaigns").select("name").lte("start_date", today).gte("end_date", today).execute().data
    active_names = [c['name'] for c in active_camps]
    
    memory = []
    # High sensitivity for better matching
    orb = cv2.ORB_create(nfeatures=2000)
    
    try:
        files = supabase.storage.from_("references").list()
        for f in files:
            if f['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = f['name'].split('_')
                if len(parts) > 0:
                    camp_name = parts[0]
                    # Only load if campaign is active TODAY
                    if camp_name in active_names:
                        url = supabase.storage.from_("references").get_public_url(f['name'])
                        resp = requests.get(url)
                        arr = np.frombuffer(resp.content, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None:
                            # 1. RESIZE REFERENCE (Standardize to 800px width)
                            h, w = img.shape
                            if w > 800: 
                                scale = 800 / w
                                img = cv2.resize(img, (800, int(h * scale)))
                            
                            # 2. LIGHTING CORRECTION (CLAHE)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            img = clahe.apply(img)
                            
                            kp, des = orb.detectAndCompute(img, None)
                            if des is not None:
                                memory.append({"campaign": camp_name, "descriptors": des})
        return memory
    except Exception as e:
        print(f"AI Memory Error: {e}")
        return []

def run_smart_audit(evidence_url, memory):
    if not memory: return 0, "Fail", "Unknown", "No Active Campaigns"
    try:
        # Download Evidence
        resp = requests.get(evidence_url)
        arr = np.frombuffer(resp.content, np.uint8)
        img_ev = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        
        # 1. RESIZE EVIDENCE (Must match reference scale)
        h, w = img_ev.shape
        if w > 800: 
            scale = 800 / w
            img_ev = cv2.resize(img_ev, (800, int(h * scale)))
        
        # 2. LIGHTING CORRECTION (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_ev = clahe.apply(img_ev)
        
        # Detect Features
        orb = cv2.ORB_create(nfeatures=2000)
        kp_ev, des_ev = orb.detectAndCompute(img_ev, None)
        
        if des_ev is None: return 0, "Fail", "Unknown", "Blurry/No Features"
        
        # 3. ROBUST MATCHING (KNN + Lowe's Ratio Test)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        best_score = 0
        best_campaign = "Unknown"
        
        for ref in memory:
            try:
                # Find top 2 matches for each point
                matches = bf.knnMatch(des_ev, ref['descriptors'], k=2)
                
                # Apply Ratio Test (filters out 90% of noise)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                score = len(good_matches)
                
                if score > best_score:
                    best_score = score
                    best_campaign = ref['campaign']
            except:
                continue
        
        # 4. FINAL DECISION
        # Threshold > 10 distinct geometric matches
        if best_score > 10: 
            return best_score, "Pass", best_campaign, "Matched Reference Pattern"
        else: 
            return best_score, "Fail", "Unknown", f"Low Match ({best_score})"

    except Exception as e:
        return 0, "Fail", "Error", str(e)

# --- 3. STORE PORTAL ---
def store_login_view():
    st.markdown("### 🏪 Store Login")
    store_code_input = st.text_input("Enter Store Code (Capital Letters)", max_chars=10).upper()
    if st.button("Login"):
        res = supabase.table("stores").select("*").eq("store_code", store_code_input).execute().data
        if res:
            st.session_state['store_user'] = res[0]
            st.rerun()
        else:
            st.error("Invalid Store Code.")

def store_upload_view():
    store = st.session_state['store_user']
    
    # Header
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"#### 👋 {store['store_name']} ({store['store_code']})")
    if c2.button("Logout"):
        st.session_state['store_user'] = None
        st.rerun()

    # Session State for Flow Control
    if 'upload_stage' not in st.session_state: st.session_state.upload_stage = "capture"

    # 1. CAPTURE STAGE
    if st.session_state.upload_stage == "capture":
        st.info("Step 1: Take or Upload a Picture")
        
        t1, t2 = st.tabs(["📷
