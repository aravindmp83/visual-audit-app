import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import cv2
import numpy as np
import requests
import time

# --- 1. SETUP ---
st.set_page_config(page_title="Retail Visual Audit", page_icon="🏢", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = init_connection()

ELEMENT_TYPES = ["Totem Pole", "Flag Pole", "Hoarding", "Facade", "Window Display"]
STATUS_OPTIONS = ["Intact", "Flex Damage", "Frame Damage", "Total Damage"]

# --- 2. AI & MEMORY FUNCTIONS ---

@st.cache_resource
def load_reference_memory():
    # Only load ACTIVE campaigns
    active_camps = supabase.table("campaigns").select("name").eq("is_active", True).execute().data
    active_names = [c['name'] for c in active_camps]
    
    memory = []
    orb = cv2.ORB_create(nfeatures=1000)
    try:
        files = supabase.storage.from_("references").list()
        for f in files:
            if f['name'].endswith(('.jpg', '.png')):
                camp_name = f['name'].split('_')[0]
                if camp_name in active_names:
                    url = supabase.storage.from_("references").get_public_url(f['name'])
                    resp = requests.get(url)
                    arr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        kp, des = orb.detectAndCompute(img, None)
                        if des is not None:
                            memory.append({"campaign": camp_name, "descriptors": des, "filename": f['name']})
        return memory
    except:
        return []

def run_smart_audit(evidence_url, memory):
    if not memory: return 0, "Fail", "Unknown", "No Active References"
    try:
        resp = requests.get(evidence_url)
        img_ev = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create(nfeatures=1000)
        kp_ev, des_ev = orb.detectAndCompute(img_ev, None)
        if des_ev is None: return 0, "Fail", "Unknown", "Blurry"
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best_score = 0
        best_campaign = "Unknown"
        
        for ref in memory:
            matches = bf.match(des_ev, ref['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            score = len([m for m in matches if m.distance < 60])
            if score > best_score:
                best_score = score
                best_campaign = ref['campaign']
        
        if best_score > 20: return best_score, "Pass", best_campaign, "Matched"
        else: return best_score, "Fail", "Unknown", "Low Score"
    except: return 0, "Error", "Error", "System Error"

# --- 3. STORE MANAGER VIEW (NO LOGIN REQUIRED) ---
def store_app_view():
    st.header("📱 Store Visual Upload Portal")
    st.info("No login required. Please locate your store below.")

    # Fetch Data
    all_stores = supabase.table("stores").select("*").execute().data
    
    # 1. Region Filter
    regions = sorted(list(set([s['region'] for s in all_stores if s['region']])))
    sel_region = st.selectbox("Select Region", regions)
    
    # 2. Cluster Filter (Filtered by Region)
    cluster_stores = [s for s in all_stores if s['region'] == sel_region]
    clusters = sorted(list(set([s['cluster_manager'] for s in cluster_stores if s['cluster_manager']])))
    sel_cluster = st.selectbox("Select Cluster Manager", clusters)
    
    # 3. Store Filter (Filtered by Cluster)
    final_stores = [s for s in cluster_stores if s['cluster_manager'] == sel_cluster]
    store_map = {f"{s['store_code']} - {s.get('store_name', '')}": s['store_code'] for s in final_stores}
    
    sel_store_display = st.selectbox("Select Store", list(store_map.keys()))
    
    if sel_store_display:
        sel_store_code = store_map[sel_store_display]
        st.divider()
        st.write(f"Uploading for: **{sel_store_code}**")
        
        element = st.selectbox("Visual Element", ELEMENT_TYPES)
        condition = st.selectbox("Condition", STATUS_OPTIONS)
        photo = st.camera_input("Take Photo")
        
        if photo:
            with st.spinner("Uploading Evidence..."):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{sel_store_code}/{element.replace(' ','')}_{timestamp}.jpg"
                file_bytes = photo.getvalue()
                
                # Upload
                supabase.storage.from_("evidence-photos").upload(path=filename, file=file_bytes, file_options={"content-type": "image/jpeg"})
                public_url = f"{st.secrets['supabase']['url']}/storage/v1/object/public/evidence-photos/{filename}"
                
                # Log
                data = {
                    "store_code": sel_store_code, "element_type": element, 
                    "condition_status": condition, "image_url": public_url, 
                    "ai_status": "Pending", "campaign_name": "Pending Scan"
                }
                supabase.table("audit_logs").insert(data).execute()
                st.success("✅ Uploaded Successfully!")

# --- 4. ADMIN DASHBOARD (LOGIN REQUIRED) ---
def admin_dashboard(user):
    st.title(f"Admin Panel ({user['role']})")
    
    # TABS: Audit vs Train AI
    tab1, tab2 = st.tabs(["🔍 Audit Dashboard", "🎓 Train AI (Campaigns)"])

    # --- TAB 1: AUDIT ---
    with tab1:
        # User Management (HQ ONLY)
        if user['role'] == 'HQ':
            with st.expander("🛠 User Management (HQ Only)"):
                st.write("Manage Admin Access")
                c1, c2 = st.columns(2)
                with c1:
                    new_user = st.text_input("New Admin Username")
                    new_pass = st.text_input("New Password", type="password")
                    new_role = st.selectbox("Role", ["Regional_Admin", "HQ"])
                    if new_role == "Regional_Admin":
                        reg = st.text_input("Region (Must match Store Data)")
                    else:
                        reg = "HQ"
                    
                    if st.button("Create User"):
                        supabase.table("users").insert({
                            "username": new_user, "password": new_pass, 
                            "role": new_role, "region": reg, "is_approved": True
                        }).execute()
                        st.success("User Created")

        # Filters
        view_mode = st.radio("View Mode", ["Status View", "Store View"], horizontal=True)
        query = supabase.table("audit_logs").select("*").order("created_at", desc=True)
        
        # Region Lock
        if user['role'] == 'Regional_Admin':
            region_stores = supabase.table("stores").select("store_code").eq("region", user['region']).execute().data
            valid_codes = [s['store_code'] for s in region_stores]
            query = query.in_("store_code", valid_codes)
        
        logs = query.execute().data
        
        if view_mode == "Status View":
            status_filter = st.selectbox("Filter Status", ["Pending", "Pass", "Fail"])
            filtered_logs = [l for l in logs if l['ai_status'] == status_filter]
            
            for log in filtered_logs:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1: st.image(log['image_url'], width=100)
                    with c2: 
                        st.write(f"**{log['store_code']}** - {log['element_type']}")
                        st.caption(f"Campaign: {log.get('campaign_name', 'Unknown')}")
                    with c3:
                        if log['ai_status'] == "Pending":
                            if st.button("Run AI", key=log['id']):
                                memory = load_reference_memory()
                                sc, stt, cp, _ = run_smart_audit(log['image_url'], memory)
                                supabase.table("audit_logs").update({"ai_score": sc, "ai_status": stt, "campaign_name": cp}).eq("id", log['id']).execute()
                                st.rerun()

        elif view_mode == "Store View":
            # Hierarchy Filter for Admin
            all_stores = supabase.table("stores").select("*").execute().data
            if user['role'] == 'Regional_Admin':
                all_stores = [s for s in all_stores if s['region'] == user['region']]
            
            clusters = sorted(list(set([s['cluster_manager'] for s in all_stores if s['cluster_manager']])))
            sel_clust = st.selectbox("Cluster", ["All"] + clusters)
            
            if sel_clust != "All":
                valid_stores = [s['store_code'] for s in all_stores if s['cluster_manager'] == sel_clust]
                logs = [l for l in logs if l['store_code'] in valid_stores]
            
            # Grid
            cols = st.columns(4)
            for idx, log in enumerate(logs):
                with cols[idx % 4]:
                    st.image(log['image_url'], use_column_width=True)
                    st.caption(f"{log['element_type']} ({log['ai_status']})")

    # --- TAB 2: TRAIN AI (Available to HQ & Regional) ---
    with tab2:
        st.subheader("Campaign Management")
        st.info("Upload official reference images here to teach the AI.")
        
        # 1. Manage Active Campaigns
        camps = supabase.table("campaigns").select("*").execute().data
        col1, col2 = st.columns(2)
        with col1:
            new_camp = st.text_input("Create New Campaign")
            if st.button("Add Campaign"):
                try:
                    supabase.table("campaigns").insert({"name": new_camp, "is_active": True}).execute()
                    st.success(f"Created {new_camp}")
                    st.rerun()
                except: st.error("Exists")
        
        with col2:
            st.write("Active Status")
            for c in camps:
                is_active = st.checkbox(c['name'], value=c['is_active'], key=c['id'])
                if is_active != c['is_active']:
                    supabase.table("campaigns").update({"is_active": is_active}).eq("id", c['id']).execute()
                    st.rerun()

        st.divider()
        # 2. Upload Reference Images
        sel_camp_upload = st.selectbox("Select Campaign for Images", [c['name'] for c in camps if c['is_active']])
        ref_files = st.file_uploader("Upload Reference JPEGs", accept_multiple_files=True)
        if st.button("Upload References"):
            if ref_files:
                for f in ref_files:
                    # Name format: CampaignName_Timestamp.jpg
                    fname = f"{sel_camp_upload}_{int(time.time())}_{f.name}"
                    fbytes = f.getvalue()
                    supabase.storage.from_("references").upload(path=fname, file=fbytes, file_options={"content-type": "image/jpeg"})
                st.success("AI Training Updated!")
                st.cache_resource.clear() # Reset memory

# --- 5. AUTH & NAVIGATION ---
def main():
    if 'user' not in st.session_state:
        st.session_state['user'] = None

    st.sidebar.image("https://img.icons8.com/color/96/shop.png", width=50)
    
    # MAIN SWITCH
    app_mode = st.sidebar.radio("Select Portal", ["Store Upload Tool", "Admin Login"])

    if app_mode == "Store Upload Tool":
        store_app_view()
        
    elif app_mode == "Admin Login":
        if st.session_state['user'] is None:
            # LOGIN FORM
            st.subheader("Admin Login")
            user_input = st.text_input("Username")
            pass_input = st.text_input("Password", type="password")
            if st.button("Login"):
                user = supabase.table("users").select("*").eq("username", user_input).eq("password", pass_input).execute().data
                if user:
                    if user[0]['is_approved']:
                        st.session_state['user'] = user[0]
                        st.rerun()
                    else: st.error("Account Pending Approval")
                else: st.error("Invalid Credentials")
        else:
            # LOGGED IN
            user = st.session_state['user']
            st.sidebar.success(f"Logged in: {user['username']}")
            if st.sidebar.button("Logout"):
                st.session_state['user'] = None
                st.rerun()
            
            admin_dashboard(user)

if __name__ == "__main__":
    main()
