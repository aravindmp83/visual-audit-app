import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import cv2
import numpy as np
import requests
import time
import pandas as pd

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Retail Visual Audit", page_icon="🏢", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = init_connection()

# Constants [cite: 7]
ELEMENT_TYPES = ["Totem Pole", "Backlit Board", "Facade", "Flagpole/Lollypop", "Others"]
STATUS_OPTIONS = ["Good", "Flex Damage", "Frame Damage", "Total Damage", "Letter Damage"]

# --- 2. AI ENGINE ---
@st.cache_resource
def load_reference_memory():
    """Loads signatures of ACTIVE campaigns only."""
    # [cite: 40] Do not auto-change status on load; just read DB.
    active_camps = supabase.table("campaigns").select("name").eq("is_active", True).execute().data
    active_names = [c['name'] for c in active_camps]
    
    memory = []
    orb = cv2.ORB_create(nfeatures=1000)
    try:
        files = supabase.storage.from_("references").list()
        for f in files:
            if f['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = f['name'].split('_')
                if len(parts) > 0:
                    camp_name = parts[0]
                    # Only load if campaign is currently active
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
    if not memory: return 0, "Fail", "Unknown", "No Active References Loaded"
    try:
        resp = requests.get(evidence_url)
        img_ev = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        orb = cv2.ORB_create(nfeatures=1000)
        kp_ev, des_ev = orb.detectAndCompute(img_ev, None)
        if des_ev is None: return 0, "Fail", "Unknown", "Image Blur / No Features"
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best_score = 0
        best_campaign = "Unknown"
        
        for ref in memory:
            matches = bf.match(des_ev, ref['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            score = len([m for m in matches if m.distance < 50])
            if score > best_score:
                best_score = score
                best_campaign = ref['campaign']
        
        # Threshold Logic
        if best_score > 15: 
            return best_score, "Pass", best_campaign, "Visual Matched Reference"
        else: 
            return best_score, "Fail", "Unknown", f"Low Match Score ({best_score})"
    except Exception as e:
        return 0, "Fail", "Error", str(e)

# --- 3. STORE PORTAL [cite: 6] ---
def store_upload_view():
    st.markdown("### 🏪 Store Visual Upload Portal")
    st.info("No login required. Select your store details below.")

    # Session State for "Upload Another" Loop [cite: 8]
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = []

    all_stores = supabase.table("stores").select("*").execute().data
    if not all_stores:
        st.error("No store data found. Contact Admin.")
        return

    # Filters [cite: 23-26]
    c1, c2 = st.columns(2)
    with c1:
        regions = sorted(list(set([s['region'] for s in all_stores if s['region']])))
        sel_region = st.selectbox("Region", regions)
        
        cluster_stores = [s for s in all_stores if s['region'] == sel_region]
        clusters = sorted(list(set([s['cluster_manager'] for s in cluster_stores if s['cluster_manager']])))
        sel_cluster = st.selectbox("Cluster Manager", clusters)

    with c2:
        final_stores = [s for s in cluster_stores if s['cluster_manager'] == sel_cluster]
        # Display Store Code - Name [cite: 29]
        store_map = {f"{s['store_code']} - {s.get('store_name', '')}": s for s in final_stores}
        sel_store_display = st.selectbox("Select Store", list(store_map.keys()))

    if sel_store_display:
        store_data = store_map[sel_store_display]
        st.divider()
        st.markdown(f"#### 📸 Uploading for: :blue[{store_data['store_code']}]")

        col_input, col_cam = st.columns([1, 2])
        with col_input:
            element = st.selectbox("Visual Element", ELEMENT_TYPES)
            condition = st.selectbox("Condition", STATUS_OPTIONS)
            
        with col_cam:
            # Camera key increments to force reset after upload
            photo = st.camera_input("Take Photo", key=f"cam_{st.session_state.uploader_key}")

        if photo:
            with st.spinner("Syncing..."):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{store_data['store_code']}/{element.replace(' ','')}_{timestamp}.jpg"
                    file_bytes = photo.getvalue()
                    
                    supabase.storage.from_("evidence-photos").upload(
                        path=filename, file=file_bytes, file_options={"content-type": "image/jpeg"}
                    )
                    public_url = f"{st.secrets['supabase']['url']}/storage/v1/object/public/evidence-photos/{filename}"
                    
                    data = {
                        "store_code": store_data['store_code'],
                        "element_type": element,
                        "condition_status": condition,
                        "image_url": public_url,
                        "ai_status": "Pending",
                        "campaign_name": "Pending Scan"
                    }
                    supabase.table("audit_logs").insert(data).execute()
                    
                    st.success(f"✅ Uploaded: {element}")
                    st.session_state.upload_history.insert(0, f"{datetime.now().strftime('%H:%M')} - {element}")
                    
                    # [cite: 8] Auto-reset camera for next element
                    st.session_state.uploader_key += 1
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        if st.session_state.upload_history:
            st.caption("Session Uploads:")
            for item in st.session_state.upload_history:
                st.write(f"✅ {item}")
            if st.button("End Session"): # [cite: 9]
                st.session_state.upload_history = []
                st.rerun()

# --- 4. AUDIT DASHBOARD [cite: 17] ---
def audit_dashboard(user_role, user_region=None):
    st.subheader("📊 Audit Dashboard")
    
    # Filters [cite: 23-26]
    c1, c2, c3, c4 = st.columns(4)
    with c1: status_view = st.selectbox("Status", ["All", "Pending", "Pass", "Fail"]) # [cite: 18-22]
    
    all_stores = supabase.table("stores").select("*").execute().data
    if user_role == 'RMM' and user_region:
        all_stores = [s for s in all_stores if s['region'] == user_region]
    
    regions = ["All"] + sorted(list(set([s['region'] for s in all_stores if s['region']])))
    with c2: sel_region = st.selectbox("Region", regions)
    
    filtered_stores_1 = all_stores if sel_region == "All" else [s for s in all_stores if s['region'] == sel_region]
    clusters = ["All"] + sorted(list(set([s['cluster_manager'] for s in filtered_stores_1 if s['cluster_manager']])))
    with c3: sel_cluster = st.selectbox("Cluster", clusters)
        
    filtered_stores_2 = filtered_stores_1 if sel_cluster == "All" else [s for s in filtered_stores_1 if s['cluster_manager'] == sel_cluster]
    # [cite: 26] All + Individual
    store_opts = ["All"] + [s['store_code'] for s in filtered_stores_2]
    with c4: sel_store = st.selectbox("Store", store_opts)

    # Query
    query = supabase.table("audit_logs").select("*").order("created_at", desc=True)
    valid_codes = [s['store_code'] for s in filtered_stores_2]
    if sel_store != "All": valid_codes = [sel_store]
    
    if not valid_codes:
        st.warning("No stores found.")
        return

    query = query.in_("store_code", valid_codes)
    if status_view != "All": query = query.eq("ai_status", status_view)
    logs = query.execute().data

    # [cite: 36] Bulk Audit
    if st.button("🚀 Run Bulk Audit"):
        with st.status("Auditing..."):
            memory = load_reference_memory()
            count = 0
            for log in logs:
                if log['ai_status'] == "Pending":
                    score, status, camp, reason = run_smart_audit(log['image_url'], memory)
                    # [cite: 34] Check if matched campaign is active
                    camp_stat = "Unknown"
                    if status == "Pass":
                        c_check = supabase.table("campaigns").select("is_active").eq("name", camp).execute().data
                        if c_check: camp_stat = "Active" if c_check[0]['is_active'] else "Inactive"
                    
                    supabase.table("audit_logs").update({
                        "ai_score": score, "ai_status": status, 
                        "campaign_name": camp, "failure_reason": reason,
                        "campaign_status": camp_stat
                    }).eq("id", log['id']).execute()
                    count += 1
            st.success(f"Audited {count} items.")
            time.sleep(1)
            st.rerun()

    # Report Card View [cite: 27]
    st.write(f"Showing {len(logs)} records")
    store_name_map = {s['store_code']: s.get('store_name', 'Unknown') for s in all_stores}

    for log in logs:
        with st.container(border=True):
            col_img, col_det, col_res = st.columns([1, 2, 1])
            with col_img: st.image(log['image_url'], width=150)
            with col_det:
                st_name = store_name_map.get(log['store_code'], '')
                st.markdown(f"**{log['store_code']} - {st_name}**") # [cite: 29]
                st.caption(f"Element: {log['element_type']}") # [cite: 30]
                st.caption(f"Condition: {log['condition_status']}") # [cite: 31]
            with col_res:
                # [cite: 32] Run Single Audit
                if log['ai_status'] == 'Pending':
                    if st.button("Run Audit", key=log['id']):
                        memory = load_reference_memory()
                        score, status, camp, reason = run_smart_audit(log['image_url'], memory)
                        
                        camp_stat = "Unknown"
                        if status == "Pass":
                            c_check = supabase.table("campaigns").select("is_active").eq("name", camp).execute().data
                            if c_check: camp_stat = "Active" if c_check[0]['is_active'] else "Inactive"

                        supabase.table("audit_logs").update({
                            "ai_score": score, "ai_status": status, 
                            "campaign_name": camp, "failure_reason": reason,
                            "campaign_status": camp_stat
                        }).eq("id", log['id']).execute()
                        st.rerun()
                else:
                    color = "green" if log['ai_status'] == "Pass" else "red"
                    st.markdown(f":{color}[**{log['ai_status']}**]")
                    st.caption(f"Camp: {log.get('campaign_name')} ({log.get('campaign_status', 'Unknown')})") # [cite: 33, 34]
                    if log['ai_status'] == "Fail": st.caption(f"Reason: {log.get('failure_reason')}") # [cite: 35]

# --- 5. TRAIN AI [cite: 37] ---
def train_ai_view():
    st.subheader("🎓 Campaign Management")
    
    t1, t2 = st.tabs(["Manage Campaigns", "Upload References"])
    
    with t1: # [cite: 38]
        with st.form("new_camp"):
            c1, c2 = st.columns([3, 1])
            new_camp = c1.text_input("New Campaign Name") # [cite: 39]
            new_stat = c2.checkbox("Active", value=True)
            if st.form_submit_button("Add"):
                try:
                    supabase.table("campaigns").insert({"name": new_camp, "is_active": new_stat}).execute()
                    st.success("Added.")
                    st.rerun()
                except: st.error("Exists.")

        # [cite: 40] Table View
        camps = supabase.table("campaigns").select("*").order("id").execute().data
        for c in camps:
            c1, c2, c3 = st.columns([1, 4, 2])
            c1.write(c['id'])
            c2.write(c['name'])
            # Direct database update on toggle
            is_active = c3.checkbox("Active", value=c['is_active'], key=f"c_{c['id']}")
            if is_active != c['is_active']:
                supabase.table("campaigns").update({"is_active": is_active}).eq("id", c['id']).execute()
                st.toast("Updated")
                time.sleep(0.5)
                st.rerun()

    with t2: # [cite: 41]
        active_camps = [c['name'] for c in camps if c['is_active']]
        sel_camp = st.selectbox("Select Campaign", active_camps) # [cite: 42]
        
        #  File Uploader
        uploaded_files = st.file_uploader("Upload JPEGs", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
        
        if st.button("Upload & Clear"):
            if uploaded_files:
                for f in uploaded_files:
                    safe_name = f.name.replace(" ", "_")
                    full_name = f"{sel_camp}_{int(time.time())}_{safe_name}"
                    try:
                        fb = f.getvalue()
                        supabase.storage.from_("references").upload(path=full_name, file=fb, file_options={"content-type": "image/jpeg"})
                    except: pass
                
                st.success("Uploaded!")
                st.cache_resource.clear()
                #  Rerun to clear the file uploader widget
                time.sleep(1)
                st.rerun()

# --- 6. AUTH & MAIN ---
def login_page(role): # [cite: 10, 14]
    st.markdown(f"### {role} Login")
    
    if role == "RMM":
        t1, t2 = st.tabs(["Login", "Sign Up"])
        with t1:
            u = st.text_input("User", key="r_u")
            p = st.text_input("Pass", type="password", key="r_p")
            if st.button("Login", key="r_btn"):
                user = supabase.table("users").select("*").eq("username", u).eq("password", p).eq("role", "RMM").execute().data
                if user:
                    if user[0]['is_approved']:
                        st.session_state['user'] = user[0]
                        st.rerun()
                    else: st.warning("Pending Approval")
                else: st.error("Invalid")
        with t2:
            nu = st.text_input("New User")
            np = st.text_input("New Pass", type="password")
            # Fetch regions dynamically
            regions = supabase.table("stores").select("region").execute().data
            reg_list = sorted(list(set([r['region'] for r in regions if r['region']])))
            nr = st.selectbox("Region", reg_list)
            if st.button("Sign Up"):
                try:
                    supabase.table("users").insert({"username": nu, "password": np, "role": "RMM", "region": nr}).execute()
                    st.success("Request Sent.")
                except: st.error("Taken.")

    else: # Admin [cite: 14]
        u = st.text_input("User", key="a_u")
        p = st.text_input("Pass", type="password", key="a_p")
        if st.button("Login", key="a_btn"):
            # [cite: 15] Admin credentials checked against table
            user = supabase.table("users").select("*").eq("username", u).eq("password", p).eq("role", "HQ").execute().data
            if user:
                st.session_state['user'] = user[0]
                st.rerun()
            else: st.error("Invalid")

def main():
    if 'user' not in st.session_state: st.session_state['user'] = None

    # [cite: 2] Home Page Dropdown
    portal = st.sidebar.selectbox("Select Portal", ["Store Picture Upload", "RMM Login", "Admin Login"])

    if st.session_state['user']:
        user = st.session_state['user']
        st.sidebar.success(f"User: {user['username']}")
        if st.sidebar.button("Logout"):
            st.session_state['user'] = None
            st.rerun()
            
        t1, t2, t3 = st.tabs(["Audit Dashboard", "Train AI", "User Mgmt"])
        with t1: audit_dashboard(user['role'], user.get('region')) # [cite: 12]
        with t2: train_ai_view() # [cite: 13]
        with t3: 
            if user['role'] == "HQ": # [cite: 16]
                st.write("Approve RMMs")
                pending = supabase.table("users").select("*").eq("is_approved", False).execute().data
                for p in pending:
                    c1, c2 = st.columns([3,1])
                    c1.write(f"{p['username']} ({p['region']})")
                    if c2.button("Approve", key=p['username']):
                        supabase.table("users").update({"is_approved": True}).eq("username", p['username']).execute()
                        st.rerun()
            else: st.info("HQ Only")
    else:
        if portal == "Store Picture Upload": store_upload_view() # [cite: 6]
        elif portal == "RMM Login": login_page("RMM") # [cite: 4]
        elif portal == "Admin Login": login_page("Admin") # [cite: 5]

if __name__ == "__main__":
    main()
