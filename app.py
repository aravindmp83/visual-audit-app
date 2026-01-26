import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import cv2
import numpy as np
import requests
import time
import pandas as pd

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Retail Visual Audit", page_icon="🏢", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = init_connection()

# Dropdown Constants
ELEMENT_TYPES = ["Totem Pole", "Flag Pole", "Hoarding", "Facade", "Others", "Lollypop"]
STATUS_OPTIONS = ["Good", "Flex Damage", "Frame Damage", "Total Damage", "Letter Damage"]

# --- 2. AI ENGINE ---
@st.cache_resource
def load_reference_memory():
    """Loads visual signatures of ACTIVE campaigns."""
    active_camps = supabase.table("campaigns").select("name").eq("is_active", True).execute().data
    active_names = [c['name'] for c in active_camps]
    
    memory = []
    orb = cv2.ORB_create(nfeatures=1000)
    try:
        files = supabase.storage.from_("references").list()
        for f in files:
            if f['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
                # Filename format: CampaignName_Timestamp_OriginalName.jpg
                parts = f['name'].split('_')
                if len(parts) > 0:
                    camp_name = parts[0]
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
            # Strict match: distance < 50
            score = len([m for m in matches if m.distance < 50])
            if score > best_score:
                best_score = score
                best_campaign = ref['campaign']
        
        # Threshold: 15 good feature matches
        if best_score > 15: 
            return best_score, "Pass", best_campaign, "Visual Matched Reference"
        else: 
            return best_score, "Fail", "Unknown", f"Low Match Score ({best_score})"
    except Exception as e:
        return 0, "Fail", "Error", str(e)

# --- 3. VIEWS ---

def store_upload_view():
    st.markdown("### 🏪 Store Visual Upload Portal")
    st.info("No login required. Select your store details below.")

    # --- 1. SESSION STATE INITIALIZATION ---
    # This keeps track of the 'refresh' for the camera and the history of uploads
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = []

    # Fetch Store Data
    all_stores = supabase.table("stores").select("*").execute().data
    if not all_stores:
        st.error("No store data found. Please contact Admin.")
        return

    # --- 2. STORE SELECTION (Persists across uploads) ---
    # We use columns to make it compact
    c1, c2 = st.columns(2)
    
    with c1:
        regions = sorted(list(set([s['region'] for s in all_stores if s['region']])))
        sel_region = st.selectbox("Region", regions)
        
        cluster_stores = [s for s in all_stores if s['region'] == sel_region]
        clusters = sorted(list(set([s['cluster_manager'] for s in cluster_stores if s['cluster_manager']])))
        sel_cluster = st.selectbox("Cluster Manager", clusters)

    with c2:
        final_stores = [s for s in cluster_stores if s['cluster_manager'] == sel_cluster]
        store_map = {f"{s['store_code']} - {s.get('store_name', '')}": s for s in final_stores}
        sel_store_display = st.selectbox("Select Store", list(store_map.keys()))

    if sel_store_display:
        store_data = store_map[sel_store_display]
        st.divider()
        st.markdown(f"#### 📸 Uploading for: :blue[{store_data['store_code']}]")

        # --- 3. THE UPLOAD FORM ---
        col_input, col_cam = st.columns([1, 2])
        
        with col_input:
            element = st.selectbox("Visual Element", ELEMENT_TYPES)
            condition = st.selectbox("Condition", STATUS_OPTIONS)
            
        with col_cam:
            # KEY TRICK: We use st.session_state.uploader_key as the key.
            # When we increment this number, Streamlit thinks it's a "new" widget and resets it.
            photo = st.camera_input("Take Photo", key=f"cam_{st.session_state.uploader_key}")

        if photo:
            with st.spinner("Syncing to Server..."):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{store_data['store_code']}/{element.replace(' ','')}_{timestamp}.jpg"
                    file_bytes = photo.getvalue()
                    
                    # 1. Upload to Storage
                    supabase.storage.from_("evidence-photos").upload(
                        path=filename, 
                        file=file_bytes, 
                        file_options={"content-type": "image/jpeg"}
                    )
                    
                    public_url = f"{st.secrets['supabase']['url']}/storage/v1/object/public/evidence-photos/{filename}"
                    
                    # 2. Log to Database
                    data = {
                        "store_code": store_data['store_code'],
                        "element_type": element,
                        "condition_status": condition,
                        "image_url": public_url,
                        "ai_status": "Pending",
                        "campaign_name": "Pending Scan"
                    }
                    supabase.table("audit_logs").insert(data).execute()
                    
                    # 3. Success Logic
                    st.success(f"✅ Uploaded: {element}")
                    
                    # Add to history list for display
                    st.session_state.upload_history.insert(0, {
                        "time": datetime.now().strftime("%H:%M"),
                        "element": element,
                        "status": "Uploaded"
                    })
                    
                    # 4. RESET THE CAMERA (The Magic Step)
                    st.session_state.uploader_key += 1
                    time.sleep(1) # Small pause so user sees the success message
                    st.rerun() # Forces app to reload with new empty camera
                    
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        # --- 4. SESSION HISTORY (So they know what they finished) ---
        if st.session_state.upload_history:
            st.divider()
            st.caption("Session Upload History (Cleared on refresh)")
            # specific visual for history
            for item in st.session_state.upload_history:
                st.markdown(f"**{item['time']}** | {item['element']} ............ ✅ **Done**")

def audit_dashboard(user_role, user_region=None):
    st.subheader("📊 Audit Dashboard")
    
    # --- FILTERS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        status_view = st.selectbox("Status View", ["All", "Pending", "Pass", "Fail"])
    
    # Store Filters Logic
    all_stores = supabase.table("stores").select("*").execute().data
    # If RMM, filter stores by their region
    if user_role == 'RMM' and user_region:
        all_stores = [s for s in all_stores if s['region'] == user_region]
    
    regions = ["All"] + sorted(list(set([s['region'] for s in all_stores if s['region']])))
    
    with c2: 
        sel_region = st.selectbox("Region", regions)
    
    # Dynamic Cluster Filter
    filtered_stores_1 = all_stores if sel_region == "All" else [s for s in all_stores if s['region'] == sel_region]
    clusters = ["All"] + sorted(list(set([s['cluster_manager'] for s in filtered_stores_1 if s['cluster_manager']])))
    
    with c3: 
        sel_cluster = st.selectbox("Cluster Manager", clusters)
        
    # Dynamic Store Filter
    filtered_stores_2 = filtered_stores_1 if sel_cluster == "All" else [s for s in filtered_stores_1 if s['cluster_manager'] == sel_cluster]
    store_opts = ["All"] + [s['store_code'] for s in filtered_stores_2]
    
    with c4: 
        sel_store = st.selectbox("Stores", store_opts)

    # --- QUERY DATA ---
    query = supabase.table("audit_logs").select("*").order("created_at", desc=True)
    
    # Apply Store Filters
    valid_store_codes = [s['store_code'] for s in filtered_stores_2]
    if sel_store != "All":
        valid_store_codes = [sel_store]
    
    if not valid_store_codes:
        st.warning("No stores found for selected filters.")
        return

    query = query.in_("store_code", valid_store_codes)
    
    # Apply Status Filter
    if status_view != "All":
        query = query.eq("ai_status", status_view)
        
    logs = query.execute().data

    # --- ACTION BAR ---
    if st.button("🚀 Run Bulk Audit on Filtered Items"):
        with st.status("Running AI Audit..."):
            memory = load_reference_memory()
            count = 0
            for log in logs:
                if log['ai_status'] == "Pending": # Only audit pending
                    score, status, camp, reason = run_smart_audit(log['image_url'], memory)
                    # Update DB
                    supabase.table("audit_logs").update({
                        "ai_score": score, "ai_status": status, 
                        "campaign_name": camp, "failure_reason": reason,
                        "campaign_status": "Active" if status == "Pass" else "Unknown"
                    }).eq("id", log['id']).execute()
                    count += 1
            st.success(f"Completed {count} audits!")
            time.sleep(1)
            st.rerun()

    # --- REPORTING CARD VIEW ---
    st.write(f"Showing {len(logs)} records")
    
    # Create a map for Store Names (to show Name instead of just Code)
    store_name_map = {s['store_code']: s.get('store_name', 'Unknown') for s in all_stores}

    for log in logs:
        with st.container(border=True):
            col_img, col_details, col_result = st.columns([1, 2, 1])
            
            with col_img:
                st.image(log['image_url'], width=150)
            
            with col_details:
                st_name = store_name_map.get(log['store_code'], '')
                st.markdown(f"**{log['store_code']} - {st_name}**")
                st.write(f"Element: **{log['element_type']}**")
                st.write(f"Condition: **{log['condition_status']}**")
                
            with col_result:
                # Status Logic
                if log['ai_status'] == 'Pending':
                    st.info("AI Status: Pending")
                    if st.button("Run Audit", key=f"btn_{log['id']}"):
                        memory = load_reference_memory()
                        score, status, camp, reason = run_smart_audit(log['image_url'], memory)
                        supabase.table("audit_logs").update({
                            "ai_score": score, "ai_status": status, 
                            "campaign_name": camp, "failure_reason": reason,
                            "campaign_status": "Active" if status == "Pass" else "Unknown"
                        }).eq("id", log['id']).execute()
                        st.rerun()
                elif log['ai_status'] == 'Pass':
                    st.success(f"PASS ✅\n\n**Campaign:** {log.get('campaign_name')}\n\n**Status:** Active")
                else:
                    st.error(f"FAIL ❌\n\n**Reason:** {log.get('failure_reason')}")

def train_ai_view():
    st.subheader("🎓 Train AI & Campaign Management")
    
    tab_camp, tab_upload = st.tabs(["Campaign Manager", "Upload References"])
    
    # --- CAMPAIGN MANAGER ---
    with tab_camp:
        # Add Campaign
        with st.form("add_campaign"):
            c1, c2 = st.columns([3, 1])
            new_camp = c1.text_input("New Campaign Name")
            new_status = c2.checkbox("Active Immediately", value=True)
            if st.form_submit_button("Add Campaign"):
                try:
                    supabase.table("campaigns").insert({"name": new_camp, "is_active": new_status}).execute()
                    st.success(f"Added {new_camp}")
                    st.rerun()
                except: st.error("Campaign name already exists")
        
        # List Campaigns (Table View with Toggle)
        st.write("Existing Campaigns:")
        camps = supabase.table("campaigns").select("*").order("id").execute().data
        
        # Header
        h1, h2, h3 = st.columns([1, 3, 2])
        h1.write("**ID**")
        h2.write("**Campaign Name**")
        h3.write("**Status (Toggle to Change)**")
        
        for c in camps:
            r1, r2, r3 = st.columns([1, 3, 2])
            r1.write(str(c['id']))
            r2.write(c['name'])
            # Status Persist Logic: "value" comes from DB
            is_active = r3.checkbox("Active", value=c['is_active'], key=f"toggle_{c['id']}")
            
            if is_active != c['is_active']:
                supabase.table("campaigns").update({"is_active": is_active}).eq("id", c['id']).execute()
                st.toast(f"Updated {c['name']} status")
                time.sleep(0.5)
                st.rerun()

    # --- UPLOAD REFERENCES ---
    with tab_upload:
        st.info("Upload official JPEGs here. Files will be renamed to avoid duplicates.")
        
        # Only show Active campaigns in dropdown
        active_camps = [c['name'] for c in camps if c['is_active']]
        sel_camp = st.selectbox("Select Campaign", active_camps)
        
        # File Uploader
        uploaded_files = st.file_uploader("Browse JPEGs", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
        
        if st.button("Upload Reference Images"):
            if uploaded_files:
                count = 0
                for f in uploaded_files:
                    # Clean filename logic
                    safe_name = f.name.replace(" ", "_")
                    # Format: Campaign_Timestamp_Name.jpg
                    full_name = f"{sel_camp}_{int(time.time())}_{safe_name}"
                    
                    try:
                        file_bytes = f.getvalue()
                        supabase.storage.from_("references").upload(path=full_name, file=file_bytes, file_options={"content-type": "image/jpeg"})
                        count += 1
                    except Exception as e:
                        st.warning(f"Skipped {f.name} (Duplicate or Error)")
                
                if count > 0:
                    st.success(f"Successfully uploaded {count} images!")
                    st.cache_resource.clear() # Clear AI memory
                    time.sleep(2)
                    st.rerun() # This clears the file uploader widget
            else:
                st.warning("Please select files first.")

# --- 4. AUTHENTICATION ---
def login_page(role):
    st.markdown(f"### {role} Login")
    
    if role == "RMM":
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            u = st.text_input("Username", key="rmm_u")
            p = st.text_input("Password", type="password", key="rmm_p")
            if st.button("Login", key="btn_rmm_login"):
                user = supabase.table("users").select("*").eq("username", u).eq("password", p).eq("role", "RMM").execute().data
                if user:
                    if user[0]['is_approved']:
                        st.session_state['user'] = user[0]
                        st.rerun()
                    else: st.warning("Account pending HQ approval.")
                else: st.error("Invalid credentials.")
        
        with tab2:
            st.write("New RMM Request")
            new_u = st.text_input("Choose Username")
            new_p = st.text_input("Choose Password", type="password")
            
            # Fetch Regions from Stores to avoid typos
            regions = supabase.table("stores").select("region").execute().data
            reg_list = sorted(list(set([r['region'] for r in regions if r['region']])))
            new_r = st.selectbox("Assign Region", reg_list)
            
            if st.button("Sign Up"):
                try:
                    supabase.table("users").insert({"username": new_u, "password": new_p, "role": "RMM", "region": new_r, "is_approved": False}).execute()
                    st.success("Request sent to Admin.")
                except: st.error("Username taken.")

    elif role == "Admin":
        u = st.text_input("Username", key="admin_u")
        p = st.text_input("Password", type="password", key="admin_p")
        if st.button("Login", key="btn_admin_login"):
             # Role can be HQ or Admin
            user = supabase.table("users").select("*").eq("username", u).eq("password", p).eq("role", "HQ").execute().data
            if user:
                st.session_state['user'] = user[0]
                st.rerun()
            else: st.error("Invalid Admin Credentials")

# --- 5. MAIN NAVIGATION ---
def main():
    if 'user' not in st.session_state:
        st.session_state['user'] = None

    # Main Dropdown for Portal Selection
    portal_options = ["Store Picture Upload", "RMM Login", "Admin Login"]
    
    # If logged in, show Dashboard instead of Login
    if st.session_state['user']:
        user = st.session_state['user']
        st.sidebar.success(f"User: {user['username']} ({user['role']})")
        if st.sidebar.button("Logout"):
            st.session_state['user'] = None
            st.rerun()
        
        # Dashboard View
        tab1, tab2, tab3 = st.tabs(["Audit Dashboard", "Train AI", "User Mgmt (HQ)"])
        
        with tab1:
            audit_dashboard(user['role'], user.get('region'))
        
        with tab2:
            train_ai_view()
            
        with tab3:
            if user['role'] == 'HQ':
                st.write("Pending Approvals")
                pending = supabase.table("users").select("*").eq("is_approved", False).execute().data
                for p in pending:
                    c1, c2 = st.columns([3, 1])
                    c1.write(f"{p['username']} ({p['region']})")
                    if c2.button("Approve", key=p['username']):
                        supabase.table("users").update({"is_approved": True}).eq("username", p['username']).execute()
                        st.rerun()
            else:
                st.info("Restricted to HQ.")

    else:
        # Not Logged In - Show Landing Page Dropdown
        choice = st.selectbox("Select Portal", portal_options)
        st.divider()
        
        if choice == "Store Picture Upload":
            store_upload_view()
        elif choice == "RMM Login":
            login_page("RMM")
        elif choice == "Admin Login":
            login_page("Admin")

if __name__ == "__main__":
    main()
