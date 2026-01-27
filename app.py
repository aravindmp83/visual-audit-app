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

# --- 2. AI ENGINE ---
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
                    # Only load if campaign is active TODAY
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
        
        if des_ev is None: return 0, "Fail", "Unknown", "Blurry/No Features"
        
        # 3. ROBUST MATCHING (KNN + Lowe's Ratio Test)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING) 
        best_score = 0
        best_campaign = "Unknown"
        
        for ref in memory:
            try:
                # Find top 2 matches for each point
                matches = bf.knnMatch(des_ev, ref['descriptors'], k=2)
                
                # Apply Ratio Test (filters out 90% of false positives)
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
        # If we have >10 solid, geometrically verified matches, it's a pass
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
        
        t1, t2 = st.tabs(["Live Camera", "File Upload"])
        photo = None
        
        with t1:
            cam = st.camera_input("Take Picture", key=f"cam_{st.session_state.get('uploader_key', 0)}")
            if cam: photo = cam
        with t2:
            # File uploader allows native camera on mobile
            up = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key=f"up_{st.session_state.get('uploader_key', 0)}")
            if up: photo = up

        if photo:
            st.session_state.temp_photo = photo
            st.session_state.upload_stage = "confirm"
            st.rerun()

    # 2. CONFIRMATION STAGE
    elif st.session_state.upload_stage == "confirm":
        st.info("Step 2: Verify & Confirm")
        
        col_img, col_form = st.columns([1, 1])
        
        with col_img:
            if st.session_state.temp_photo:
                st.image(st.session_state.temp_photo, caption="Captured Image", use_column_width=True)
            if st.button("❌ Retake"):
                st.session_state.temp_photo = None
                st.session_state.upload_stage = "capture"
                st.rerun()

        with col_form:
            st.markdown("### Details")
            ele = st.selectbox("Visual Element", ELEMENT_TYPES)
            cond = st.selectbox("Condition", STATUS_OPTIONS)
            
            st.divider()
            
            if st.button("✅ Confirm & Upload", type="primary", use_container_width=True):
                with st.spinner("Uploading..."):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{store['store_code']}/{ele.replace(' ','')}_{timestamp}.jpg"
                        file_bytes = st.session_state.temp_photo.getvalue()
                        
                        supabase.storage.from_("evidence-photos").upload(filename, file_bytes, {"content-type": "image/jpeg"})
                        public_url = f"{st.secrets['supabase']['url']}/storage/v1/object/public/evidence-photos/{filename}"
                        
                        supabase.table("audit_logs").insert({
                            "store_code": store['store_code'],
                            "element_type": ele,
                            "condition_status": cond,
                            "image_url": public_url,
                            "ai_status": "Pending",
                            "campaign_name": "Pending Scan"
                        }).execute()
                        
                        st.success("Uploaded!")
                        st.session_state.upload_stage = "next_question"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # 3. NEXT ELEMENT QUESTION
    elif st.session_state.upload_stage == "next_question":
        st.balloons()
        st.markdown("### 🌟 Upload Successful!")
        st.markdown("#### Do you have another marketing element in this store?")
        
        c_yes, c_no = st.columns(2)
        
        if c_yes.button("✅ Yes, Upload Next", use_container_width=True):
            st.session_state.uploader_key = st.session_state.get('uploader_key', 0) + 1
            st.session_state.temp_photo = None
            st.session_state.upload_stage = "capture"
            st.rerun()
            
        if c_no.button("❌ No, Finish", use_container_width=True):
            st.session_state['store_user'] = None
            st.session_state.upload_stage = "capture"
            st.success("Logging out...")
            time.sleep(1)
            st.rerun()

# --- 4. AUDIT DASHBOARD ---
def audit_dashboard(user_role, user_region=None):
    st.subheader("📊 Audit Dashboard")
    
    # Filters
    c1, c2, c3, c4 = st.columns(4)
    with c1: status_view = st.selectbox("Status", ["All", "Pending", "Pass", "Fail"])
    
    all_stores = supabase.table("stores").select("*").execute().data
    if user_role == 'RMM' and user_region:
        all_stores = [s for s in all_stores if s['region'] == user_region]
    
    regions = ["All"] + sorted(list(set([s['region'] for s in all_stores if s['region']])))
    with c2: sel_region = st.selectbox("Region", regions)
    
    filtered_1 = all_stores if sel_region == "All" else [s for s in all_stores if s['region'] == sel_region]
    clusters = ["All"] + sorted(list(set([s['cluster_manager'] for s in filtered_1 if s['cluster_manager']])))
    with c3: sel_clust = st.selectbox("Cluster", clusters)
    
    filtered_2 = filtered_1 if sel_clust == "All" else [s for s in filtered_1 if s['cluster_manager'] == sel_clust]
    store_opts = ["All"] + [s['store_code'] for s in filtered_2]
    with c4: sel_store = st.selectbox("Store", store_opts)

    # Query
    query = supabase.table("audit_logs").select("*").order("created_at", desc=True)
    if sel_store != "All": query = query.eq("store_code", sel_store)
    elif user_role == 'RMM': 
        valid_codes = [s['store_code'] for s in all_stores]
        query = query.in_("store_code", valid_codes)
        
    if status_view != "All": query = query.eq("ai_status", status_view)
    logs = query.execute().data

    # Bulk Audit
    if st.button("🚀 Run Bulk Audit"):
        with st.status("Auditing..."):
            memory = load_reference_memory()
            count = 0
            for log in logs:
                if log['ai_status'] == "Pending":
                    score, status, camp, reason = run_smart_audit(log['image_url'], memory)
                    
                    camp_stat = "Inactive"
                    if status == "Pass":
                        today = date.today().isoformat()
                        c = supabase.table("campaigns").select("*").eq("name", camp).lte("start_date", today).gte("end_date", today).execute().data
                        if c: camp_stat = "Active"
                    
                    supabase.table("audit_logs").update({
                        "ai_score": score, "ai_status": status, 
                        "campaign_name": camp, "failure_reason": reason,
                        "campaign_status": camp_stat
                    }).eq("id", log['id']).execute()
                    count += 1
            st.success(f"Processed {count} images.")
            time.sleep(1)
            st.rerun()

    # Cards
    for log in logs:
        with st.container(border=True):
            col_img, col_info, col_act = st.columns([1, 2, 1])
            with col_img: st.image(log['image_url'], width=150)
            with col_info:
                st.write(f"**{log['store_code']}** - {log['element_type']}")
                st.caption(f"Condition: {log['condition_status']}")
                if log['ai_status'] != "Pending":
                    st.caption(f"Campaign: {log.get('campaign_name')} ({log.get('campaign_status','Unknown')})")
            with col_act:
                if log['ai_status'] == "Pending":
                    if st.button("Run Audit", key=log['id']):
                        memory = load_reference_memory()
                        score, status, camp, reason = run_smart_audit(log['image_url'], memory)
                        
                        camp_stat = "Inactive"
                        if status == "Pass":
                            today = date.today().isoformat()
                            c = supabase.table("campaigns").select("*").eq("name", camp).lte("start_date", today).gte("end_date", today).execute().data
                            if c: camp_stat = "Active"

                        supabase.table("audit_logs").update({
                            "ai_score": score, "ai_status": status, 
                            "campaign_name": camp, "failure_reason": reason,
                            "campaign_status": camp_stat
                        }).eq("id", log['id']).execute()
                        st.rerun()
                else:
                    color = "green" if log['ai_status'] == "Pass" else "red"
                    st.markdown(f":{color}[**{log['ai_status']}**]")
                    if log['ai_status'] == "Fail": st.caption(log.get('failure_reason'))

# --- 5. CAMPAIGN MANAGEMENT ---
def train_ai_view():
    st.subheader("🎓 Campaign Management")
    t1, t2 = st.tabs(["Manage Campaigns", "Upload References"])
    
    with t1:
        with st.form("new_camp"):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Campaign Name")
            s_date = c2.date_input("Start Date")
            e_date = c3.date_input("End Date")
            if st.form_submit_button("Add Campaign"):
                try:
                    supabase.table("campaigns").insert({
                        "name":name, "start_date": s_date.isoformat(), "end_date": e_date.isoformat()
                    }).execute()
                    st.success("Added")
                    st.rerun()
                except: st.error("Error/Duplicate")
        
        camps = supabase.table("campaigns").select("*").execute().data
        if camps:
            df = pd.DataFrame(camps)
            st.dataframe(df[['name', 'start_date', 'end_date']], use_container_width=True)

    with t2:
        camp_names = [c['name'] for c in camps] if camps else []
        sel_camp = st.selectbox("Select Campaign", camp_names)
        uploaded = st.file_uploader("Upload JPEGs", accept_multiple_files=True)
        if st.button("Upload Reference"):
            if uploaded:
                for f in uploaded:
                    safe_name = f.name.replace(" ", "_")
                    fname = f"{sel_camp}_{safe_name}"
                    try:
                        fb = f.getvalue()
                        supabase.storage.from_("references").upload(fname, fb, {"content-type": "image/jpeg"})
                    except: pass
                st.success("Completed")
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()

# --- 6. AUTH & MAIN ---
def login_page(role):
    st.markdown(f"### {role} Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    
    if st.button("Login"):
        r_check = "HQ" if role == "Admin" else "RMM"
        user = supabase.table("users").select("*").eq("username", u).eq("password", p).eq("role", r_check).execute().data
        if user:
            if user[0]['is_approved']:
                st.session_state['user'] = user[0]
                st.rerun()
            else: st.warning("Pending Approval")
        else: st.error("Invalid Credentials")

    if role == "RMM":
        with st.expander("New RMM? Sign Up"):
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            regions = supabase.table("stores").select("region").execute().data
            reg_list = sorted(list(set([r['region'] for r in regions if r['region']])))
            nr = st.selectbox("Region", reg_list)
            if st.button("Request Access"):
                try:
                    supabase.table("users").insert({"username":nu, "password":np, "role":"RMM", "region":nr}).execute()
                    st.success("Request Sent")
                except: st.error("Username taken")

def main():
    if 'store_user' not in st.session_state: st.session_state['store_user'] = None
    if 'user' not in st.session_state: st.session_state['user'] = None

    if st.session_state['store_user']:
        store_upload_view()
    elif st.session_state['user']:
        user = st.session_state['user']
        st.sidebar.write(f"User: {user['username']}")
        if st.sidebar.button("Logout"):
            st.session_state['user'] = None
            st.rerun()
        
        t1, t2, t3 = st.tabs(["Audit Dashboard", "Train AI", "Admin"])
        with t1: audit_dashboard(user['role'], user.get('region'))
        with t2: train_ai_view()
        with t3:
            if user['role'] == "HQ":
                st.write("Approve RMMs")
                pending = supabase.table("users").select("*").eq("is_approved", False).execute().data
                for p in pending:
                    c1, c2 = st.columns([3,1])
                    c1.write(f"{p['username']} ({p['region']})")
                    if c2.button("Approve", key=p['username']):
                        supabase.table("users").update({"is_approved": True}).eq("username", p['username']).execute()
                        st.rerun()
            else: st.info("Restricted")

    else:
        choice = st.selectbox("Select Portal", ["Store Login", "RMM Login", "Admin Login"])
        if choice == "Store Login": store_login_view()
        elif choice == "RMM Login": login_page("RMM")
        elif choice == "Admin Login": login_page("Admin")

if __name__ == "__main__":
    main()
