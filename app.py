import streamlit as st
import requests

# 1. PURE BLACK THEME CSS
BLACK_CSS = """
<style>
    /* Main Background - Pure Black */
    .stApp, [data-testid="stAppViewContainer"] { background-color: #000000 !important; }
    
    /* Sidebar - Near Black */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background-color: #0a0a0a !important; border-right: 1px solid #222222 !important; }
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label { color: #ffffff !important; }
    
    /* Chat Bubbles */
    div[data-testid="stChatMessage"] { background-color: #111111 !important; border: 1px solid #222222 !important; }
    div[data-testid="stChatMessage"] .stMarkdown p { color: #ffffff !important; }
    
    /* Chat Input Area */
    [data-testid="stChatInput"] { background-color: #000000 !important; }
    [data-testid="stChatInputTextArea"] textarea { background-color: #111111 !important; color: #ffffff !important; border-color: #333333 !important; }
    
    /* General Inputs & File Uploader */
    input, textarea { background-color: #111111 !important; color: #ffffff !important; border-color: #333333 !important; }
    [data-testid="stFileUploader"] { background-color: #0a0a0a !important; border: 1px dashed #333333 !important; }
    
    /* Buttons - Sky Blue Accent */
    button[kind="primary"], .stButton>button { background-color: #00bfff !important; color: #ffffff !important; border-radius: 8px !important; border: none !important; font-weight: 600 !important; }
    button[kind="primary"]:hover, .stButton>button:hover { background-color: #009acd !important; }
    
    /* Text Colors */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, p, span, label { color: #ffffff !important; }
    
    /* Feature Cards */
    .feature-card { 
        background-color: #111111 !important; 
        border: 1px solid #222222 !important; 
        border-radius: 12px !important; 
        padding: 15px !important; 
        transition: transform 0.2s, border-color 0.2s !important; 
    }
    .feature-card:hover { transform: translateY(-2px) !important; border-color: #00bfff !important; }
    .feature-card h3 { color: #00bfff !important; margin-top: 0 !important; }
    .feature-card p { color: #a0a0a0 !important; margin-bottom: 0 !important; }

    /* Hide default Streamlit elements */
    #MainMenu, footer { visibility: hidden !important; }
</style>
"""

# 2. PAGE CONFIG & THEME INJECTION
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

# Inject Black CSS unconditionally
st.markdown(BLACK_CSS, unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

# 3. SIDEBAR: CONTROLS ONLY (No Toggle)
with st.sidebar:
    st.header("📄 Document Control")
    uploaded_file = st.file_uploader("Upload PDF (Max 20 pages)", type=["pdf"], key="uploader")
    
    if "active_doc" not in st.session_state: 
        st.session_state.active_doc = None
        
    if st.session_state.active_doc: 
        st.success(f"✅ Active: {st.session_state.active_doc}")
    else: 
        st.info("ℹ️ No document loaded")

    st.divider()
    
    # Ingestion Logic
    if uploaded_file:
        if uploaded_file.name != st.session_state.active_doc:
            with st.status(f"Analyzing {uploaded_file.name}...", expanded=True) as status:
                st.write("📤 Uploading to backend...")
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/ingest", files=files, timeout=180)
                    if response.status_code == 200:
                        st.write(f"✅ {response.json()['message']}")
                        status.update(label="Analysis Complete", state="complete")
                        st.session_state.active_doc = uploaded_file.name
                        st.session_state.messages = [] 
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                        status.update(label="Failed", state="error")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                    status.update(label="Failed", state="error")
        else: 
            st.success("Document already indexed!")

    st.divider()
    if st.button("🗑️ New Chat / Clear History"):
        st.session_state.messages = []
        st.rerun()

# 4. MAIN AREA: HEADER & FEATURE SHOWCASE
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("🧠 DocuMind AI")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("v4.2.0 | Vision-RAG")

st.caption("Multimodal Document Analysis | Precision Extraction | Conversational Memory")
st.divider()

st.markdown("### 🚀 What DocuMind Can Perform")
f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown('<div class="feature-card"><h3>👁️ Vision Analysis</h3><p>Reads and interprets charts, tables, and diagrams directly from PDF pages.</p></div>', unsafe_allow_html=True)
with f2:
    st.markdown('<div class="feature-card"><h3>🎯 Precision Extraction</h3><p>Extracts exact numbers, dates, and metrics without conversational fluff.</p></div>', unsafe_allow_html=True)
with f3:
    st.markdown('<div class="feature-card"><h3>🧠 Conversational Memory</h3><p>Remembers previous questions for seamless, context-aware follow-ups.</p></div>', unsafe_allow_html=True)
with f4:
    st.markdown('<div class="feature-card"><h3>⚡ Lightning Fast</h3><p>Pre-processes documents at upload for instant, sub-second query responses.</p></div>', unsafe_allow_html=True)

st.divider()

# 5. CHAT INTERFACE
st.subheader("💬 Chat with your Document")

if "messages" not in st.session_state: 
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""<div style="text-align: center; color: #808080; margin-top: 30px; margin-bottom: 30px;">
        <h3>👋 Welcome to DocuMind</h3>
        <p>Upload a document on the left, then ask precise questions below.</p>
    </div>""", unsafe_allow_html=True)

# Render Chat History (STRICTLY VALID EMOJIS ONLY)
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a precise question..."):
    if not st.session_state.active_doc:
        st.error("Please upload a document first!"); st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # STRICTLY VALID EMOJI
    with st.chat_message("user", avatar="👤"): 
        st.markdown(prompt)

    # STRICTLY VALID EMOJI
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Synthesizing precise answer..."):
            try:
                history_to_send = st.session_state.messages[:-1]
                response = requests.post(
                    f"{API_URL}/query", 
                    json={"query": prompt, "chat_history": history_to_send}, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    if not answer or not answer.strip():
                        answer = "I couldn't find a precise answer for that in the document. Could you try rephrasing?"
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Backend connection failed. Is Uvicorn running?")