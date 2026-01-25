import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import cv2
import numpy as np
import requests
from PIL import Image
import io

# --- 1. SETUP ---
st.set_page_config(page_title="Retail Visual Audit", page_icon="🏢", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = init_connection()

# Constants
ELEMENT_TYPES = ["Totem Pole", "Flag Pole", "Hoarding", "Facade", "Window Display"]
STATUS_OPTIONS = ["Intact", "Flex Damage", "Frame Damage", "Total Damage"]

# --- 2. AI ENGINE (ORB) ---
def run_ai_audit(evidence_url, element_type):
    """Downloads evidence and reference, compares them using ORB."""
    try:
        # A. Download Evidence Image
        resp_ev = requests.get(evidence_url)
        img_ev = cv2.imdecode(np.frombuffer(resp_ev.content, np.uint8), cv2.IMREAD_GRAYSCALE)

        # B. Download Reference Image (from 'references' bucket)
        # Assumes filename is "TotemPole.jpg"
        ref_filename = f"{element_type.replace(' ', '')}.jpg" 
        ref_url = f"{st.secrets['supabase']['url']}/storage/v1/object/public/references/{ref_filename}"
        
        resp_ref = requests.get(ref_url)
        if resp_ref.status_code != 200:
            return 0, "Ref Missing" # Reference image not found

        img_ref = cv2.imdecode(np.frombuffer(resp_ref.content, np.uint8), cv2.IMREAD_GRAYSCALE)

        # C. ORB Matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_ev, None)
        kp2, des2 = orb.detectAndCompute(img_ref, None)
        
        if des1 is None or des2 is None: return 0, "Blurry/No Feature"

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Simple Scoring: Top 20 matches avg distance
        score = len([m for m in matches if m.distance < 50])
        final_score = min(score * 2, 100) # Cap at 100
        
        status = "Pass" if final_score > 30 else "Fail"
        return final_score, status

    except Exception as e:
        return 0, f"Error: {str(e)}"

# --- 3. INTERFACE: STORE MANAGER (UPLOAD) ---
def store_manager_view():
    st.title("📱 Store Visual Upload")
    
    # Hierarchy Logic
    try:
        # Get all stores
        resp = supabase.table("stores").select("*").execute()
        df_stores = resp.data
        
        # 1. Select Cluster
        clusters = list(set([row['cluster_manager'] for row in df_stores]))
        selected_cluster = st.selectbox("Select Cluster Manager", clusters)
        
        # 2. Filter Stores by Cluster
        filtered_stores = [row for row in df_stores if row['cluster_manager'] == selected_cluster]
        store_map = {f"{row['store_code']} ({row['store_manager']})": row['store_code'] for row in filtered_stores}
        
        selected_store_display = st.selectbox("Select Store", list(store_map.keys()))
        selected_store_code = store_map[selected_store_display]
        
        st.info(f"Welcome, {filtered_stores[0]['store_manager']}!")

        # 3. Upload Form
        element = st.selectbox("Visual Element", ELEMENT_TYPES)
        condition = st.selectbox("Condition Status", STATUS_OPTIONS)
        photo = st.camera_input("Take Photo")

        if photo:
            with st.spinner("Uploading..."):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{selected_store_code}/{element.replace(' ', '')}_{timestamp}.jpg"
                file_bytes = photo.getvalue()
                
                # Upload to Supabase Storage
                supabase.storage.from_("evidence-photos").upload(
                    path=filename, file=file_bytes, file_options={"content-type": "image/jpeg"}
                )
                
                # Get Public URL
                project_url = st.secrets["supabase"]["url"]
                public_url = f"{project_url}/storage/v1/object/public/evidence-photos/{filename}"
                
                # Save to DB
                data = {
                    "store_code": selected_store_code,
                    "element_type": element,
                    "condition_status": condition,
                    "image_url": public_url,
                    "ai_status": "Pending"
                }
                supabase.table("audit_logs").insert(data).execute()
                st.success("✅ Uploaded Successfully!")

    except Exception as e:
        st.error(f"System Error: {e}")

# --- 4. INTERFACE: HEAD OFFICE (AUDIT) ---
def head_office_view():
    st.title("💻 Central Command Center")
    
    # 1. Bulk Actions
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Visual Audit Gallery")
    with col2:
        if st.button("🚀 Run BULK AI Audit"):
            with st.status("Running AI on pending images..."):
                # Fetch pending logs
                pending = supabase.table("audit_logs").select("*").eq("ai_status", "Pending").execute().data
                if not pending:
                    st.write("No pending images to audit.")
                else:
                    for log in pending:
                        st.write(f"Auditing {log['store_code']} - {log['element_type']}...")
                        score, status = run_ai_audit(log['image_url'], log['element_type'])
                        
                        # Update DB
                        supabase.table("audit_logs").update({"ai_score": score, "ai_status": status}).eq("id", log['id']).execute()
                    st.success("Bulk Audit Complete!")
                    st.rerun()

    # 2. Filters
    # Fetch Hierarchy
    resp = supabase.table("stores").select("*").execute()
    stores_data = resp.data
    clusters = ["All"] + list(set([s['cluster_manager'] for s in stores_data]))
    
    sel_cluster = st.selectbox("Filter by Cluster", clusters)
    
    # Filter Logic
    query = supabase.table("audit_logs").select("*").order("created_at", desc=True)
    if sel_cluster != "All":
        # Find store codes for this cluster
        valid_codes = [s['store_code'] for s in stores_data if s['cluster_manager'] == sel_cluster]
        query = query.in_("store_code", valid_codes)
        
    logs = query.execute().data

    # 3. Gallery Grid
    if not logs:
        st.info("No uploads found.")
        return

    for log in logs:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c1:
                st.image(log['image_url'], use_column_width=True)
            
            with c2:
                st.write(f"**{log['store_code']}** | {log['element_type']}")
                st.caption(f"Status: {log['condition_status']}")
                st.caption(f"Uploaded: {log['created_at'][:10]}")
            
            with c3:
                # AI Status Display
                if log['ai_status'] == "Pending":
                    st.warning("AI: Pending")
                    if st.button("Run Audit", key=f"btn_{log['id']}"):
                        score, status = run_ai_audit(log['image_url'], log['element_type'])
                        supabase.table("audit_logs").update({"ai_score": score, "ai_status": status}).eq("id", log['id']).execute()
                        st.rerun()
                elif log['ai_status'] == "Pass":
                    st.success(f"AI: PASS ({log['ai_score']}%)")
                else:
                    st.error(f"AI: FAIL ({log['ai_score']}%)")

# --- 5. MAIN NAVIGATION ---
def main():
    st.sidebar.image("https://img.icons8.com/color/96/shop.png", width=50)
    mode = st.sidebar.radio("Select Interface", ["Store Manager App", "Head Office Admin"])
    
    if mode == "Store Manager App":
        store_manager_view()
    else:
        head_office_view()

if __name__ == "__main__":
    main()
