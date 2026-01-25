import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import cv2
import numpy as np
import requests
import time

# --- 1. SETUP ---
st.set_page_config(page_title="AI Visual Auditor", page_icon="🧠", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = init_connection()

# Constants
ELEMENT_TYPES = ["Totem Pole", "Flag Pole", "Hoarding", "Facade", "Window Display"]
STATUS_OPTIONS = ["Intact", "Flex Damage", "Frame Damage", "Total Damage"]

# --- 2. THE LEARNING AI ENGINE ---

@st.cache_resource
def load_reference_memory():
    """
    Downloads ALL reference images from Supabase 'references' bucket 
    and converts them to mathematical descriptors (ORB).
    This runs ONCE and stays in memory = Super Fast.
    """
    memory = []
    orb = cv2.ORB_create(nfeatures=1000)
    
    try:
        # List all files in 'references' bucket
        files = supabase.storage.from_("references").list()
        
        for f in files:
            if f['name'].endswith(('.jpg', '.jpeg', '.png')):
                # Download
                url = supabase.storage.from_("references").get_public_url(f['name'])
                resp = requests.get(url)
                arr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    kp, des = orb.detectAndCompute(img, None)
                    if des is not None:
                        # Extract Campaign Name from filename (e.g., "WinterSale_01.jpg" -> "WinterSale")
                        camp_name = f['name'].split('_')[0] 
                        memory.append({
                            "campaign": camp_name,
                            "filename": f['name'],
                            "descriptors": des
                        })
        return memory
    except Exception as e:
        st.error(f"Memory Load Error: {e}")
        return []

def run_smart_audit(evidence_url, memory):
    """Compares evidence against the entire Memory."""
    if not memory:
        return 0, "Fail", "Unknown", "No References Found"

    try:
        # Download Evidence
        resp = requests.get(evidence_url)
        img_ev = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        orb = cv2.ORB_create(nfeatures=1000)
        kp_ev, des_ev = orb.detectAndCompute(img_ev, None)
        
        if des_ev is None:
            return 0, "Fail", "Unknown", "Image too blurry / No features"

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        best_score = 0
        best_campaign = "Unknown"
        best_ref_file = ""

        # Check against ALL references in memory
        for ref in memory:
            matches = bf.match(des_ev, ref['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Score = Number of good matches (distance < 50)
            good_matches = [m for m in matches if m.distance < 60]
            score = len(good_matches)
            
            if score > best_score:
                best_score = score
                best_campaign = ref['campaign']
                best_ref_file = ref['filename']

        # LOGIC: Define Thresholds
        # Adjust '20' based on real-world testing
        if best_score > 20:
            return best_score, "Pass", best_campaign, f"Matched {best_ref_file}"
        else:
            return best_score, "Fail", "Unknown", f"Low Match Score (Best: {best_score} matches)"

    except Exception as e:
        return 0, "Error", "Error", str(e)

# --- 3. STORE MANAGER VIEW ---
def store_manager_view():
    st.title("📱 Store Visual Upload")
    
    # Simple Store Selector
    stores = supabase.table("stores").select("store_code").execute().data
    store_codes = [s['store_code'] for s in stores]
    
    sel_store = st.selectbox("Select Store", store_codes)
    element = st.selectbox("Visual Element", ELEMENT_TYPES)
    condition = st.selectbox("Condition", STATUS_OPTIONS)
    photo = st.camera_input("Take Photo")

    if photo:
        with st.spinner("Uploading..."):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{sel_store}/{element.replace(' ','')}_{timestamp}.jpg"
            file_bytes = photo.getvalue()
            
            supabase.storage.from_("evidence-photos").upload(
                path=filename, file=file_bytes, file_options={"content-type": "image/jpeg"}
            )
            
            public_url = f"{st.secrets['supabase']['url']}/storage/v1/object/public/evidence-photos/{filename}"
            
            # Save Initial Log
            data = {
                "store_code": sel_store,
                "element_type": element,
                "condition_status": condition,
                "image_url": public_url,
                "ai_status": "Pending",
                "campaign_name": "Pending Scan"
            }
            supabase.table("audit_logs").insert(data).execute()
            st.success("✅ Uploaded! HQ will audit shortly.")

# --- 4. HEAD OFFICE ADMIN (THE BRAIN) ---
def head_office_view():
    st.title("🧠 AI Command Center")
    
    # Load Memory
    memory = load_reference_memory()
    st.caption(f"AI Knowledge: {len(memory)} reference images loaded.")

    tab1, tab2 = st.tabs(["🔍 Audit Dashboard", "🎓 Train AI (References)"])

    # TAB 1: AUDIT & REVIEW
    with tab1:
        # Filter for Pending
        filter_status = st.radio("Show:", ["Pending", "Fail", "Pass"], horizontal=True)
        logs = supabase.table("audit_logs").select("*").eq("ai_status", filter_status).order("created_at", desc=True).execute().data
        
        if not logs:
            st.info("No records found.")
        
        for log in logs:
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 2, 2])
                with c1:
                    st.image(log['image_url'], width=150)
                
                with c2:
                    st.write(f"**{log['store_code']}**")
                    st.caption(f"Element: {log['element_type']}")
                    st.caption(f"Reason: {log.get('failure_reason', 'N/A')}")
                
                with c3:
                    # ACTION BUTTONS
                    if log['ai_status'] == "Pending":
                        if st.button("⚡ Run AI Scan", key=f"run_{log['id']}"):
                            score, status, camp, reason = run_smart_audit(log['image_url'], memory)
                            # Update DB
                            supabase.table("audit_logs").update({
                                "ai_score": score,
                                "ai_status": status,
                                "campaign_name": camp,
                                "failure_reason": reason
                            }).eq("id", log['id']).execute()
                            st.rerun()
                    
                    elif log['ai_status'] == "Fail":
                        st.error(f"Failed. Reason: {log.get('failure_reason')}")
                        st.write("Does this belong to a campaign?")
                        
                        # THE LEARNING DROPDOWN
                        campaigns = supabase.table("campaigns").select("name").execute().data
                        camp_list = [c['name'] for c in campaigns]
                        new_camp = st.selectbox("Assign Campaign:", camp_list, key=f"sel_{log['id']}")
                        
                        if st.button("🎓 Teach AI (Add to References)", key=f"teach_{log['id']}"):
                            # 1. Download the Evidence Image
                            resp = requests.get(log['image_url'])
                            img_bytes = resp.content
                            
                            # 2. Upload to REFERENCES bucket with Campaign Name
                            # Format: CampaignName_RandomID.jpg
                            new_ref_name = f"{new_camp}_{int(time.time())}.jpg"
                            supabase.storage.from_("references").upload(
                                path=new_ref_name, 
                                file=img_bytes, 
                                file_options={"content-type": "image/jpeg"}
                            )
                            
                            # 3. Update the Log
                            supabase.table("audit_logs").update({
                                "ai_status": "Pass",
                                "campaign_name": new_camp,
                                "failure_reason": "Manually Verified & Learned"
                            }).eq("id", log['id']).execute()
                            
                            # 4. Clear Cache to reload memory next time
                            st.cache_resource.clear()
                            st.success(f"AI Learned! '{new_ref_name}' added to references.")
                            time.sleep(1)
                            st.rerun()

    # TAB 2: CAMPAIGN MANAGER
    with tab2:
        st.subheader("Add Official Campaign Visuals")
        new_camp_name = st.text_input("Campaign Name (e.g., SummerSale)")
        ref_file = st.file_uploader("Upload Official JPEGs", accept_multiple_files=True)
        
        if st.button("Upload Reference"):
            if new_camp_name and ref_file:
                for f in ref_file:
                    file_name = f"{new_camp_name}_{f.name}"
                    file_bytes = f.getvalue()
                    supabase.storage.from_("references").upload(
                        path=file_name, file=file_bytes, file_options={"content-type": "image/jpeg"}
                    )
                # Save Campaign Name to DB if unique
                try:
                    supabase.table("campaigns").insert({"name": new_camp_name}).execute()
                except:
                    pass # Ignore duplicate names
                
                st.cache_resource.clear()
                st.success("References Updated!")

# --- 5. MAIN ---
def main():
    if "admin" not in st.session_state:
        st.session_state.admin = False

    st.sidebar.image("https://img.icons8.com/color/96/shop.png", width=50)
    mode = st.sidebar.radio("Mode", ["Store App", "HQ Admin"])
    
    if mode == "Store App":
        store_manager_view()
    else:
        head_office_view()

if __name__ == "__main__":
    main()
