import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import cv2
import numpy as np
import requests
import time

# --- 1. SETUP ---
st.set_page_config(page_title="Enterprise Visual Audit", page_icon="🔐", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = init_connection()

# Constants
ELEMENT_TYPES = ["Totem Pole", "Flag Pole", "Hoarding", "Facade", "Window Display"]
STATUS_OPTIONS = ["Intact", "Flex Damage", "Frame Damage", "Total Damage"]

# --- 2. AUTHENTICATION MODULE ---
def login():
    st.markdown("## 🔐 Secure Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Fetch user
        user = supabase.table("users").select("*").eq("username", username).eq("password", password).execute().data
        
        if user:
            u = user[0]
            if not u['is_approved']:
                st.error("Account pending approval by HQ.")
            else:
                st.session_state['user'] = u
                st.success(f"Welcome {u['role']}")
                st.rerun()
        else:
            st.error("Invalid credentials")

    st.divider()
    with st.expander("Register New Regional Admin"):
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        new_region = st.text_input("Region (e.g., South, North)")
        if st.button("Request Access"):
            try:
                supabase.table("users").insert({
                    "username": new_user, "password": new_pass, 
                    "role": "Regional_Admin", "region": new_region, "is_approved": False
                }).execute()
                st.success("Request sent to HQ for approval.")
            except:
                st.error("Username already taken.")

def logout():
    st.session_state['user'] = None
    st.rerun()

# --- 3. AI ENGINE (unchanged) ---
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
                # Only load if campaign is active
                if camp_name in active_names:
                    url = supabase.storage.from_("references").get_public_url(f['name'])
                    resp = requests.get(url)
                    arr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        kp, des = orb.detectAndCompute(img, None)
                        if des is not None:
                            memory.append({"campaign": camp_name, "descriptors": des})
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

# --- 4. VIEWS ---

def store_app_view():
    st.header("📱 Store Visual Update")
    
    # 1. Fetch Stores
    all_stores = supabase.table("stores").select("*").execute().data
    
    # 2. Cluster Filter
    clusters = sorted(list(set([s['cluster_manager'] for s in all_stores if s['cluster_manager']])))
    sel_cluster = st.selectbox("Select Cluster Manager", clusters)
    
    # 3. Store Code Filter (Nested)
    filtered_stores = [s for s in all_stores if s['cluster_manager'] == sel_cluster]
    store_options = [f"{s['store_code']} - {s.get('store_name', '')}" for s in filtered_stores]
    sel_store_display = st.selectbox("Select Store", store_options)
    
    if sel_store_display:
        sel_store_code = sel_store_display.split(" - ")[0]
        
        # 4. Upload
        element = st.selectbox("Visual Element", ELEMENT_TYPES)
        condition = st.selectbox("Condition", STATUS_OPTIONS)
        photo = st.camera_input("Take Photo")
        
        if photo:
            with st.spinner("Uploading..."):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{sel_store_code}/{element.replace(' ','')}_{timestamp}.jpg"
                file_bytes = photo.getvalue()
                
                supabase.storage.from_("evidence-photos").upload(path=filename, file=file_bytes, file_options={"content-type": "image/jpeg"})
                public_url = f"{st.secrets['supabase']['url']}/storage/v1/object/public/evidence-photos/{filename}"
                
                data = {
                    "store_code": sel_store_code, "element_type": element, 
                    "condition_status": condition, "image_url": public_url, 
                    "ai_status": "Pending", "campaign_name": "Pending Scan"
                }
                supabase.table("audit_logs").insert(data).execute()
                st.success("Uploaded!")

def admin_dashboard(user):
    st.title(f"Admin Panel ({user['region'] if user['region'] else 'HQ'})")
    
    # Security: Password Reset & User Approval (HQ Only)
    if user['role'] == 'HQ':
        with st.expander("🛠 User Management"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Pending Approvals")
                pending = supabase.table("users").select("*").eq("is_approved", False).execute().data
                for p in pending:
                    if st.button(f"Approve {p['username']}", key=p['username']):
                        supabase.table("users").update({"is_approved": True}).eq("username", p['username']).execute()
                        st.rerun()
            with c2:
                st.subheader("Reset Password")
                target_user = st.text_input("Target Username")
                new_pw = st.text_input("New Password", type="password", key="new_pw")
                if st.button("Reset"):
                    supabase.table("users").update({"password": new_pw}).eq("username", target_user).execute()
                    st.success("Password Updated")

    # --- AUDIT VIEW ---
    st.subheader("Audit Dashboard")
    
    view_mode = st.radio("View Mode", ["Status View", "Store View"], horizontal=True)
    
    # FILTER LOGIC (Region Security)
    query = supabase.table("audit_logs").select("*").order("created_at", desc=True)
    
    # If Regional Admin, filter only their stores
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
                    else:
                        st.write(f"**{log['ai_status']}**")

    elif view_mode == "Store View":
        # Cluster -> Store Dropdown
        all_stores = supabase.table("stores").select("*").execute().data
        # Filter stores if regional
        if user['role'] == 'Regional_Admin':
            all_stores = [s for s in all_stores if s['region'] == user['region']]
            
        clusters = sorted(list(set([s['cluster_manager'] for s in all_stores if s['cluster_manager']])))
        sel_clust = st.selectbox("Cluster", ["All"] + clusters)
        
        if sel_clust != "All":
            valid_stores = [s['store_code'] for s in all_stores if s['cluster_manager'] == sel_clust]
            logs = [l for l in logs if l['store_code'] in valid_stores]
            
            sel_store = st.selectbox("Store Code", ["All"] + valid_stores)
            if sel_store != "All":
                logs = [l for l in logs if l['store_code'] == sel_store]
        
        st.write(f"Showing {len(logs)} images")
        # Grid View
        cols = st.columns(4)
        for idx, log in enumerate(logs):
            with cols[idx % 4]:
                st.image(log['image_url'], use_column_width=True)
                st.caption(f"{log['element_type']} ({log['ai_status']})")

    # --- CAMPAIGN MANAGER ---
    if user['role'] == 'HQ':
        st.divider()
        st.subheader("Campaign Management")
        
        # List Campaigns
        camps = supabase.table("campaigns").select("*").execute().data
        for c in camps:
            c1, c2 = st.columns([3, 1])
            with c1: st.write(f"**{c['name']}**")
            with c2: 
                is_active = st.checkbox("Active", value=c['is_active'], key=f"camp_{c['id']}")
                if is_active != c['is_active']:
                    supabase.table("campaigns").update({"is_active": is_active}).eq("id", c['id']).execute()
                    st.toast("Updated")
                    time.sleep(1)
                    st.rerun()

# --- 5. MAIN ---
def main():
    # Session State for User
    if 'user' not in st.session_state:
        st.session_state['user'] = None

    if st.session_state['user'] is None:
        login()
    else:
        # Sidebar
        user = st.session_state['user']
        st.sidebar.write(f"Logged in as: **{user['username']}** ({user['role']})")
        if st.sidebar.button("Logout"):
            logout()
            
        if user['role'] == 'Store_User': # If you create store users
            store_app_view()
        elif user['role'] in ['HQ', 'Regional_Admin']:
            # Admin can toggle between upload view and dashboard
            mode = st.sidebar.radio("Mode", ["Dashboard", "Upload Tool"])
            if mode == "Dashboard":
                admin_dashboard(user)
            else:
                store_app_view()

if __name__ == "__main__":
    main()
