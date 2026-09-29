import os
import base64
import io
import requests
import chromadb
import streamlit as st
import fitz # PyMuPDF
from groq import Groq
from dotenv import load_dotenv

# 1. LOAD CONFIGURATION
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not NVIDIA_API_KEY:
    st.error("CRITICAL: NVIDIA_API_KEY is missing from Streamlit Secrets!")

JINA_MODEL = "jina-embeddings-v5-omni-small"
NVIDIA_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
GROQ_MODEL = "openai/gpt-oss-120b"

# 2. PIPELINE CLASSES
class JinaEmbedder:
    def __init__(self):
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}

    def embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        payload = {"model": JINA_MODEL, "input": [{"text": t} for t in texts], "task": task, "normalized": True}
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

class NvidiaVisionBridge:
    def __init__(self):
        self.url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}

    def describe_image(self, img_base64: str) -> str:
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
            raise Exception(f"NVIDIA API Error {response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class GroqGenerator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, query: str, context: str, chat_history: list) -> str:
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

# 3. GLOBAL STATE
@st.cache_resource
def init_db():
    client = chromadb.PersistentClient(path="./chroma_db_storage")
    return client.get_or_create_collection(name="documind_streamlit_cloud", metadata={"hnsw:space": "cosine"})

collection = init_db()
embedder = JinaEmbedder()
vision_bridge = NvidiaVisionBridge()
generator = GroqGenerator()

# 4. UI & LOGIC
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
    
    if uploaded_file and uploaded_file.name != st.session_state.active_doc:
        with st.status(f"Analyzing {uploaded_file.name}...", expanded=True) as status:
            try:
                file_bytes = uploaded_file.read()
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                max_pages = min(len(doc), 20)
                
                valid_texts, valid_ids, valid_metas = [], [], []
                for i in range(max_pages):
                    page = doc[i]
                    mat = fitz.Matrix(2, 2) 
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("jpeg")
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8').replace('\n', '')
                    
                    text_desc = vision_bridge.describe_image(img_b64)
                    if "SKIP_PAGE" in text_desc or len(text_desc.strip()) < 30: continue
                    
                    valid_texts.append(text_desc)
                    valid_ids.append(f"doc_{uploaded_file.name}_page_{i}")
                    valid_metas.append({"type": "text", "source": uploaded_file.name, "page": i})
                
                if valid_texts:
                    vectors = embedder.embed_batch(valid_texts, task="retrieval.passage")
                    collection.add(ids=valid_ids, embeddings=vectors, metadatas=valid_metas, documents=valid_texts)
                
                st.write(f"✅ Successfully analyzed and indexed {len(valid_texts)} pages.")
                status.update(label="Analysis Complete", state="complete")
                st.session_state.active_doc = uploaded_file.name
                st.session_state.messages = []
                doc.close()
            except Exception as e:
                st.error(f"Extraction Failed: {str(e)}")
                status.update(label="Failed", state="error")

    if st.button("🗑️ New Chat / Clear History"):
        st.session_state.messages = []
        st.rerun()

st.subheader("💬 Chat with your Document")
if "messages" not in st.session_state: st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""<div style="text-align: center; color: #808080; margin-top: 30px; margin-bottom: 30px;"><h3>👋 Welcome to DocuMind</h3><p>Upload a document on the left, then ask precise questions below.</p></div>""", unsafe_allow_html=True)

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

    # FIX 2: EXPLICIT EMOJI (NO MORE EMPTY STRINGS)
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Synthesizing precise answer..."):
            try:
                query_vector = embedder.embed_batch([prompt], task="retrieval.query")[0]
                results = collection.query(query_embeddings=[query_vector], n_results=4)
                
                if not results['ids'][0]:
                    answer = "No document context found. Please upload a PDF first."
                else:
                    context_parts = []
                    for idx, doc_text in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][idx]
                        context_parts.append(f"[Source {idx+1} - Page {metadata['page']+1}]:\n{doc_text}")
                    
                    answer = generator.generate(prompt, "\n\n".join(context_parts), st.session_state.messages[:-1])
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
