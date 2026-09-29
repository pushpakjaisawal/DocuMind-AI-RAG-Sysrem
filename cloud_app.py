import os
import base64
import io
import sys
import logging
import traceback
import requests
import chromadb
import streamlit as st
import fitz # PyMuPDF
from groq import Groq
from dotenv import load_dotenv

# 1. CONFIGURE DEBUG LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DocuMind")

# 2. LOAD CONFIGURATION & VALIDATE SECRETS
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# CRITICAL DEBUG CHECK: Ensure keys are actually loaded from Streamlit Secrets
missing_keys = []
if not JINA_API_KEY: missing_keys.append("JINA_API_KEY")
if not NVIDIA_API_KEY: missing_keys.append("NVIDIA_API_KEY")
if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY")

if missing_keys:
    st.error(f"🚨 CRITICAL: Missing API Keys in Streamlit Secrets: {', '.join(missing_keys)}")
    st.info("Go to 'Manage App' (top right) -> 'Settings' -> 'Secrets' and add them in TOML format.")
    st.stop()

logger.info("✅ All API keys successfully loaded from environment.")

JINA_MODEL = "jina-embeddings-v5-omni-small"
NVIDIA_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
# FIX: Updated to a verified, active Groq model ID
GROQ_MODEL = "llama-3.3-70b-versatile" 

# 3. PIPELINE CLASSES
class JinaEmbedder:
    def __init__(self):
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}

    def embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        logger.info(f"Embedding {len(texts)} texts via Jina (Task: {task})...")
        payload = {"model": JINA_MODEL, "input": [{"text": t} for t in texts], "task": task, "normalized": True}
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        if resp.status_code != 200:
            logger.error(f"Jina API Error: {resp.status_code} - {resp.text}")
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

class NvidiaVisionBridge:
    def __init__(self):
        self.url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}

    def describe_image(self, img_base64: str, page_num: int) -> str:
        logger.info(f"Sending Page {page_num} to NVIDIA Vision API...")
        payload = {
            "model": NVIDIA_VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": "You are an exhaustive data extraction engine. Extract ALL text, skills, technologies, qualifications, dates, and metrics VERBATIM. DO NOT summarize. If it is a list of skills or items, list EVERY SINGLE ONE. Output ONLY the extracted data. If the page is completely blank, output exactly: 'SKIP_PAGE'."}
            ]}],
            "temperature": 0.1, "max_tokens": 1500, "stream": False
        }
        response = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"NVIDIA API Error {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        response.raise_for_status()
        extracted_text = response.json()["choices"][0]["message"]["content"]
        logger.info(f"Page {page_num} extracted successfully.")
        return extracted_text

class GroqGenerator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, query: str, context: str, chat_history: list) -> str:
        logger.info(f"Generating response via Groq. Context length: {len(context)} chars.")
        system_prompt = """You are a precision AI analyst analyzing a document. 
        RULES: 1. Answer comprehensively. If the user asks for a list, provide the COMPLETE list. 
        2. Start DIRECTLY with the answer. No pleasantries. 3. Use ONLY the provided context. 
        4. If not found, reply EXACTLY: 'Data not found in the provided context.' 5. Cite sources like [Source 1]."""
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": f"Document Context:\n{context}\n\nUser Query: {query}"})

        completion = self.client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.1, max_tokens=1024)
        answer = completion.choices[0].message.content
        return answer if answer and answer.strip() else "Data not found in the provided context."

# 4. GLOBAL STATE
@st.cache_resource
def init_db():
    logger.info("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db_storage")
    return client.get_or_create_collection(name="documind_streamlit_cloud", metadata={"hnsw:space": "cosine"})

collection = init_db()
embedder = JinaEmbedder()
vision_bridge = NvidiaVisionBridge()
generator = GroqGenerator()

# 5. UI & LOGIC
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #000000 !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background-color: #0a0a0a !important; border-right: 1px solid #222222 !important; }
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label { color: #ffffff !important; }
    div[data-testid="stChatMessage"] { background-color: #111111 !important; border: 1px solid #222222 !important; }
    div[data-testid="stChatMessage"] .stMarkdown p { color: #ffffff !important; }
    [data-testid="stChatInput"] { background-color: #000000 !important; }
    [data-testid="stChatInputTextArea"] textarea { background-color: #111111 !important; color: #ffffff !important; border-color: #333333 !important; }
    input, textarea { background-color: #111111 !important; color: #ffffff !important; border-color: #333333 !important; }
    [data-testid="stFileUploader"] { background-color: #0a0a0a !important; border: 1px dashed #333333 !important; }
    button[kind="primary"], .stButton>button { background-color: #00bfff !important; color: #ffffff !important; border-radius: 8px !important; border: none !important; font-weight: 600 !important; }
    button[kind="primary"]:hover, .stButton>button:hover { background-color: #009acd !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, p, span, label { color: #ffffff !important; }
    .feature-card { background-color: #111111 !important; border: 1px solid #222222 !important; border-radius: 12px !important; padding: 15px !important; transition: transform 0.2s, border-color 0.2s !important; }
    .feature-card:hover { transform: translateY(-2px) !important; border-color: #00bfff !important; }
    .feature-card h3 { color: #00bfff !important; margin-top: 0 !important; }
    .feature-card p { color: #a0a0a0 !important; margin-bottom: 0 !important; }
    #MainMenu, footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DocuMind AI")
st.caption("Multimodal Vision-RAG | Precision Document Analysis")
st.divider()

with st.sidebar:
    st.header("📄 Document Control")
    uploaded_file = st.file_uploader("Upload PDF (Max 20 pages)", type=["pdf"], key="uploader")
    
    if "active_doc" not in st.session_state: st.session_state.active_doc = None
    if st.session_state.active_doc: st.success(f"✅ Active: {st.session_state.active_doc}")
    else: st.info("ℹ️ No document loaded")
    st.divider()
    
    # DEBUG EXPANDER
    with st.expander("🛠️ Debug / System Logs", expanded=False):
        st.caption("Live backend status:")
        try:
            # Safe count check for ChromaDB
            doc_count = collection.count() if hasattr(collection, 'count') else len(collection.get()['ids'])
            st.info(f"DB Collection Size: {doc_count} documents")
        except Exception:
            st.info("DB Collection Size: 0 documents")

    if uploaded_file and uploaded_file.name != st.session_state.active_doc:
        with st.status(f"Analyzing {uploaded_file.name}...", expanded=True) as status:
            try:
                logger.info(f"Starting ingestion for: {uploaded_file.name}")
                file_bytes = uploaded_file.read()
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                max_pages = min(len(doc), 20)
                logger.info(f"PDF loaded. Total pages to process: {max_pages}")
                
                valid_texts, valid_ids, valid_metas = [], [], []
                for i in range(max_pages):
                    page = doc[i]
                    mat = fitz.Matrix(2, 2) 
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("jpeg")
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8').replace('\n', '')
                    
                    text_desc = vision_bridge.describe_image(img_b64, page_num=i+1)
                    if "SKIP_PAGE" in text_desc or len(text_desc.strip()) < 30: 
                        logger.info(f"Skipping page {i+1} (Blank/Too short).")
                        continue
                    
                    valid_texts.append(text_desc)
                    valid_ids.append(f"doc_{uploaded_file.name}_page_{i}")
                    valid_metas.append({"type": "text", "source": uploaded_file.name, "page": i})
                
                if valid_texts:
                    vectors = embedder.embed_batch(valid_texts, task="retrieval.passage")
                    collection.add(ids=valid_ids, embeddings=vectors, metadatas=valid_metas, documents=valid_texts)
                    logger.info(f"Successfully added {len(valid_texts)} chunks to ChromaDB.")
                
                st.write(f"✅ Successfully analyzed and indexed {len(valid_texts)} pages.")
                status.update(label="Analysis Complete", state="complete")
                st.session_state.active_doc = uploaded_file.name
                st.session_state.messages = []
                doc.close()
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Ingestion Failed:\n{error_trace}")
                st.error(f"🚨 Extraction Failed: {str(e)}")
                status.update(label="Failed", state="error")

    if st.button("🗑️ New Chat / Clear History"):
        st.session_state.messages = []
        st.rerun()

st.subheader("💬 Chat with your Document")
if "messages" not in st.session_state: st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""<div style="text-align: center; color: #808080; margin-top: 30px; margin-bottom: 30px;"><h3> Welcome to DocuMind</h3><p>Upload a document on the left, then ask precise questions below.</p></div>""", unsafe_allow_html=True)

# RENDER HISTORY (STRICTLY VALID EMOJIS)
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# CHAT INPUT
if prompt := st.chat_input("Ask a precise question..."):
    if not st.session_state.active_doc:
        st.error("Please upload a document first!"); st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # FIX 1: EXPLICIT EMOJI
    with st.chat_message("user", avatar="👤"): 
        st.markdown(prompt)

    # FIX 2: EXPLICIT EMOJI
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Synthesizing precise answer..."):
            try:
                query_vector = embedder.embed_batch([prompt], task="retrieval.query")[0]
                results = collection.query(query_embeddings=[query_vector], n_results=4)
                
                if not results['ids'][0]:
                    logger.warning("No context retrieved from ChromaDB.")
                    answer = "No document context found. Please upload a PDF first."
                else:
                    logger.info(f"Retrieved {len(results['ids'][0])} chunks from DB.")
                    context_parts = []
                    for idx, doc_text in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][idx]
                        context_parts.append(f"[Source {idx+1} - Page {metadata['page']+1}]:\n{doc_text}")
                    
                    answer = generator.generate(prompt, "\n\n".join(context_parts), st.session_state.messages[:-1])
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Generation Failed:\n{error_trace}")
                st.error(f" Generation Failed: {str(e)}")
