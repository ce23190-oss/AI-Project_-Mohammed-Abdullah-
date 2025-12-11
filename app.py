import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io

# --- 1. APP CONFIGURATION ---
# This sets the page title and icon in the browser tab
st.set_page_config(page_title="Smart CV Filter", page_icon="📄", layout="wide")

# --- 2. RESET FUNCTION ---
# This allows the "Reset System" button to clear everything
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def reset_app():
    st.session_state.reset_key += 1
    st.rerun()

k = st.session_state.reset_key

# --- 3. CSS STYLING (The Visuals) ---
# This block handles the "Pearlescent" background, Black buttons, and Custom Fonts
st.markdown("""
    <style>
    /* The 10-Minute Infinite Loop Background */
    .stApp {
        background: linear-gradient(
            125deg, 
            #4a00e0, #00f2ff, #ff0099, #120c6e, #4a00e0
        );
        background-size: 300% 300%;
        animation: gradient 20s ease-in-out infinite;
        color: white;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Make Input Boxes Transparent */
    .stTextArea>div>div, .stFileUploader>div>div, .stMultiSelect>div>div, .stTextInput>div>div, .stCheckbox>div>div>div {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        backdrop-filter: blur(4px);
    }
    
    textarea, input { color: white !important; }
    
    /* Styling for Black Buttons with Cyan Border */
    div.stButton > button {
        background: #000000;
        color: white;
        border: 1px solid #00f2ff;
        padding: 12px 28px;
        border-radius: 4px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.4s;
    }
    div.stButton > button:hover {
        background: #1a1a1a; 
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.6); 
        color: #00f2ff;
        transform: translateY(-2px);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #111;
    }
    
    /* The "Winner Box" Styling */
    .winner-box {
        background: rgba(0, 0, 0, 0.6);
        border-left: 5px solid #00f2ff;
        padding: 20px;
        border-radius: 5px;
        text-align: left;
        margin-bottom: 20px;
    }
    
    /* Skill Badges (Chips) */
    .skill-chip {
        background: #000000;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        margin: 0 5px 0 0;
        border: 1px solid rgba(255,255,255,0.3);
        display: inline-block;
    }
    
    h1, h2, h3 { color: #f0f0f0; font-family: sans-serif; font-weight: 300; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR (Control Panel) ---
with st.sidebar:
    st.title("CONTROL PANEL")
    st.markdown("---")
    
    top_n = st.slider("🏆 Candidates to Show", 1, 50, 10, key=f"top_n_{k}")
    min_score = st.slider("🎯 Cut-off Score", 0, 100, 10, key=f"min_score_{k}")
    
    st.markdown("### ⚡ SKILL BOOSTERS")
    must_have_skills = st.multiselect(
        "Required Skills (+10 / -10):",
        ["Python", "Java", "C++", "SQL", "React", "AWS", "Docker", "Linux"],
        default=[],
        key=f"skills_{k}"
    )
    custom_skill = st.text_input("Add Custom Skill:", placeholder="Type & Enter", key=f"custom_skill_{k}")
    
    # The "God Mode" Button
    max_priority = st.checkbox("🔥 Max Priority (Strict Filter)", key=f"priority_{k}", 
                               help="If checked, candidates who DO NOT have the custom skill will be deleted.")
    
    st.markdown("### 🛑 Block List")
    blocked_words = st.multiselect(
        "Immediately exclude if found:",
        ["Intern", "Student", "Junior", "Freelance"],
        default=[],
        key=f"blocked_{k}"
    )
    custom_block = st.text_input("Custom Block Word:", placeholder="Type & Enter", key=f"custom_block_{k}")
    
    st.markdown("---")
    if st.button("♻️ RESET SYSTEM"):
        reset_app()

# --- 5. MAIN INTERFACE ---
st.title("Smart CV Filter: AI Edition") 
st.markdown("### Intelligent Candidate Ranking System")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 1️⃣ Job Specs")
    job_description = st.text_area(
        "Job Specs", 
        height=200, 
        placeholder="Enter engine specs... (I mean, Job Description)",
        label_visibility="collapsed",
        key=f"job_desc_{k}"
    )

with col2:
    st.markdown("#### 2️⃣ Upload Data")
    uploaded_files = st.file_uploader(
        "Upload", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{k}"
    )

st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    process_btn = st.button("🚀 INITIALIZE SCAN", key=f"btn_{k}")

# --- 6. HELPER FUNCTION: EXTRACT TEXT ---
def extract_text(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages])
    except:
        return ""

# --- 7. MAIN LOGIC (The "AI" Part) ---
if process_btn:
    if not job_description or not uploaded_files:
        st.error("⚠️ SYSTEM ALERT: Please provide Job Specs and Files.")
    else:
        with st.spinner("⚙️ PROCESSING DATA STREAMS..."):
            cv_data = []
            
            # Combine standard list and custom list
            required_skills = must_have_skills.copy()
            if custom_skill: 
                # Smart Split: Turns "Rome, Java" into ["Rome", "Java"]
                for s in custom_skill.split(','):
                    if s.strip():
                        required_skills.append(s.strip())
            
            all_blocked = blocked_words.copy()
            if custom_block: all_blocked.append(custom_block)
            
            # --- LOOP THROUGH EACH PDF ---
            for file in uploaded_files:
                text = extract_text(file)
                if len(text) > 20:
                    
                    # A. CHECK MAX PRIORITY (Strict Filter)
                    # If enabled, delete anyone who doesn't have the custom skills
                    if max_priority:
                        # Extract just the custom skills from the sidebar input
                        custom_only = []
                        if custom_skill:
                            for s in custom_skill.split(','):
                                if s.strip(): custom_only.append(s.strip())
                        
                        if custom_only:
                            found_priority = False
                            for cs in custom_only:
                                if cs.lower() in text.lower():
                                    found_priority = True
                                    break
                            
                            if not found_priority:
                                continue # Skip this candidate (Delete)
                            
                    skill_bonus = 0
                    matched_skills = []
                    
                    # B. CALCULATE SKILL SCORE (+10 / -10)
                    for skill in required_skills:
                        if skill.lower() in text.lower():
                            skill_bonus += 10 # Reward
                            matched_skills.append(skill)
                        else:
                            skill_bonus -= 10 # Penalty
                            
                    cv_data.append({
                        "Name": file.name, 
                        "Text": text, 
                        "Bonus": skill_bonus,
                        "Skills Found": ", ".join(matched_skills)
                    })
            
            if cv_data:
                df = pd.DataFrame(cv_data)
                
                # --- C. THE AI ENGINE (TF-IDF & Cosine Similarity) ---
                # 1. Prepare data for the AI
                all_texts = [job_description] + df['Text'].tolist()
                
                # 2. Convert Text to Numbers (Vectors)
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                
                # 3. Measure Similarity (Distance between Job Spec and CV)
                matches = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
                
                # 4. Save the Base Score (0-100)
                df['Base Score'] = [round(score * 100, 2) for score in matches[0]]
                
                # --- D. FINAL SCORE CALCULATION ---
                # Final = AI Score + Skill Bonus
                df['Final Score'] = df['Base Score'] + df['Bonus']
                
                # Ensure score stays between 0 and 100
                df['Final Score'] = df['Final Score'].clip(0, 100)
                
                # Convert to pure Integer (No decimals)
                df['Final Score'] = df['Final Score'].astype(int)
                
                # --- E. BLOCK LIST FILTER ---
                if all_blocked:
                    for word in all_blocked:
                        # If a bad word is found, remove the candidate row
                        df = df[~df['Text'].str.contains(word, case=False)]
                
                # Filter by Cut-off Score
                df = df[df['Final Score'] >= min_score]
                
                # Sort by Highest Score
                df = df.sort_values(by='Final Score', ascending=False).head(top_n).reset_index(drop=True)
                
                st.markdown("---")
                
                # --- F. DISPLAY RESULTS ---
                if not df.empty:
                    top_c = df.iloc[0]
                    clean_name = top_c['Name'].replace(".pdf", "").replace(".docx", "")
                    
                    skills_html = ""
                    if top_c['Skills Found']:
                        for s in top_c['Skills Found'].split(', '):
                            skills_html += f"<span class='skill-chip'>{s}</span>"
                    else:
                        skills_html = "<span style='color:#ccc; font-style:italic;'>No specific skills matched</span>"
                    
                    st.markdown(f"""
                        <div class="winner-box">
                            <h2 style="margin:0">🏆 Top Match: {clean_name}</h2>
                            <h1 style="font-size: 50px; color: #ffffff;">{top_c['Final Score']}% MATCH</h1>
                            <p style="margin-top: 10px;">{skills_html}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Excel Download Logic (No Text Column)
                    buffer = io.BytesIO()
                    export_df = df[['Name', 'Final Score', 'Skills Found']]
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, index=False, sheet_name='Ranked Candidates')
                        
                    st.download_button(
                        label="📥 Download Top Candidates (Excel)",
                        data=buffer,
                        file_name="ranked_candidates.xlsx",
                        mime="application/vnd.ms-excel",
                    )
                    
                    st.dataframe(
                        df[['Name', 'Final Score', 'Skills Found']].style.background_gradient(cmap="Blues"),
                        use_container_width=True
                    )
                    
                    # Chart Logic
                    fig, ax = plt.subplots(figsize=(6, 2))
                    fig.patch.set_alpha(0) 
                    ax.patch.set_alpha(0)
                    
                    ax.barh(df['Name'], df['Final Score'], color='#000000')
                    ax.set_yticks([]) # Hide names on left
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
                    
                else:
                    st.warning("⚠️ No candidates matched the parameters.")