import streamlit as st
from supabase import create_client, Client
from datetime import datetime, date
import google.generativeai as genai
import requests
import time
import pandas as pd
from PIL import Image
import io

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Retail Visual Audit", page_icon="🏢", layout="wide")

@st.cache_resource
def init_connections():
    try:
        # Supabase
        su_url = st.secrets["supabase"]["url"]
        su_key = st.secrets["supabase"]["key"]
        db = create_client(su_url, su_key)
        
        # Google Gemini (Get Key: https://aistudio.google.com/)
        genai.configure(api_key=st.secrets["google"]["api_key"])
        
        return db
    except:
        return None

supabase: Client = init_connections()

ELEMENT_TYPES = ["Totem Pole", "Backlit Board", "Flagpole/Lollypop", "Façade", "Select your signage", "Others"]
STATUS_OPTIONS = ["Intact", "Flex Damage", "Frame Damage", "Total Damage", "Letter Damage"]

# --- 2. AI ENGINE (OPTIMIZED BATCHING) ---
def get_active_campaign_references():
    """Fetches reference images for campaigns active TODAY."""
    today = date.today().isoformat()
    # 1. Get Active Campaigns
    active_camps = supabase.table("campaigns").select("name").lte("start_date", today).gte("end_date", today).execute().data
    active_names = [c['name'] for c in active_camps]
    
    if not active_names: return []

    # 2. Find matching files in storage
    references = []
    try:
        files = supabase.storage.from_("references").list()
        for f in files:
            parts = f['name'].split('_')
            # Check if file belongs to an active campaign
            if len(parts) > 0 and parts[0] in active_names:
                url = supabase.storage.from_("references").get_public_url(f['name'])
                references.append({"name": parts[0], "url": url})
    except:
        pass
    return references

def run_gemini_audit(evidence_url, references):
    if not references: return 0, "Fail", "Unknown", "No Active Campaigns"
    
    try:
        # 1. Prepare Evidence Image
        resp_ev = requests.get(evidence_url)
        img_ev = Image.open(io.BytesIO(resp_ev.content))

        # 2. Prepare Reference Images (Batch Download)
        ref_images = []
        ref_names = []
        for ref in references:
            try:
                r = requests.get(ref['url'])
                img = Image.open(io.BytesIO(r.content))
                ref_images.append(img)
                ref_names.append(ref['name'])
            except: continue

        if not ref_images: return 0, "Fail", "Unknown", "References failed to load"

        # 3. Construct the "Batch" Prompt
        # We send ALL references at once to save API quota
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = [
            "Act as a Retail Audit AI.",
            "I will provide you with ONE 'Store Photo' and a list of 'Reference Creatives'.",
            "Your Task: Identify if the 'Store Photo' contains ANY of the 'Reference Creatives'.",
            "Rules:",
            "1. Analyze the 'Store Photo' (the last image provided).",
            "2. Compare it against the Reference images provided earlier.",
            "3. Ignore minor lighting/angle differences.",
            "4. If a match is found, return the exact Name of the reference.",
            "5. If NO match is found, return 'Fail'.",
            "Output format: Status | ConfidenceScore | MatchedName | Reason",
            "Example: Pass | 95 | WinterSale | Exact match found on totem pole.",
            "Here are the Reference Creatives:",
            *ref_images,  # Unpacks all reference images
            f"Reference Names corresponding to images: {', '.join(ref_names)}",
            "Here is the Store Photo:",
            img_ev
        ]
        
        # 4. Run AI
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # 5. Parse Response
        # Expected: Pass | 90 | CampaignName | Reason
        parts = text.split('|')
        if len(parts) >= 3:
            status = parts[0].strip()
            score = int(parts[1].strip())
            camp = parts[2].strip()
            reason = parts[3].strip() if len(parts) > 3 else "AI Match"
            
            if status.lower() == "pass":
                return score, "Pass", camp, reason
        
        return 0, "Fail", "Unknown", "No match found in batch analysis"

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
        
        t1, t2 = st.tabs(["📷 Live Camera", "📂 File Upload"])
        photo = None
        
        with t1:
            cam = st.camera_input("Take Picture", key=f"cam_{st.session_state.get('uploader_key', 0)}")
            if cam: photo = cam
        with t2:
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
    st.subheader("📊 Audit Dashboard (Gemini AI)")
    
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
    if st.button("🚀 Run Gemini Audit"):
        with st.status("Thinking... (Powered by Google Gemini)"):
            references = get_active_campaign_references()
            count = 0
            for log in logs:
                if log['ai_status'] == "Pending":
                    # Uses the BATCH function now
                    score, status, camp, reason = run_gemini_audit(log['image_url'], references)
                    
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
                        references = get_active_campaign_references()
                        score, status, camp, reason = run_gemini_audit(log['image_url'], references)
                        
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
