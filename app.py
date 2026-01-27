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

# --- 2. AI ENGINE (UPDATED: LIGHTING & SCALE FIX) ---
@st.cache_resource
def load_reference_memory():
    # Fetch active campaigns based on date
    today = date.today().isoformat()
    # Logic: Start Date <= Today <= End Date
    active_camps = supabase.table("campaigns").select("name").lte("start_date", today).gte("end_date", today).execute().data
    active_names = [c['name'] for c in active_camps]
    
    memory = []
    # Increased sensitivity (2000 features)
    orb = cv2.ORB_create(nfeatures=2000)
    
    try:
        files = supabase.storage.from_("references").list()
        for f in files:
            if f['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = f['name'].split('_')
                if len(parts) > 0:
                    camp_name = parts[0]
                    if camp_name in active_names:
                        url = supabase.storage.from_("references").get_public_url(f['name'])
                        resp = requests.get(url)
                        arr = np.frombuffer(resp.content, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None:
                            # 1. RESIZE REFERENCE (Crucial for consistency)
                            h, w = img.shape
                            if w > 800: img = cv2.resize(img, (800, int(h*(800/w))))
                            
                            # 2. LIGHTING CORRECTION (CLAHE)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            img = clahe.apply(img)
                            
                            kp, des = orb.detectAndCompute(img, None)
                            if des is not None:
                                memory.append({"campaign": camp_name, "descriptors": des})
        return memory
    except Exception as e:
        print(f"Memory Error: {e}")
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
        if w > 800: img_ev = cv2.resize(img_ev, (800, int(h*(800/w))))
        
        # 2. LIGHTING CORRECTION (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_ev = clahe.apply(img_ev)
        
        # Detect Features
        orb = cv2.ORB_create(nfeatures=2000)
        kp_ev, des_ev = orb.detectAndCompute(img_ev, None)
        
        if des_ev is None: return 0, "Fail", "Unknown", "Blur
