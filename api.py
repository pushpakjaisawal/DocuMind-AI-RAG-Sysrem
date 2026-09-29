import os
import base64
import io
import time
import requests
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# 1. LOAD CONFIGURATION
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

POPPLER_PATH = None 

JINA_MODEL = "jina-embeddings-v5-omni-small"
NVIDIA_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
GROQ_MODEL = "openai/gpt-oss-120b"

app = FastAPI(title="DocuMind AI API", version="4.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []

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
        # FIX 1: EXHAUSTIVE EXTRACTION PROMPT
        payload = {
            "model": NVIDIA_VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": "You are an exhaustive data extraction engine. Extract ALL text, skills, technologies, qualifications, dates, and metrics VERBATIM. DO NOT summarize. If it is a list of skills or items, list EVERY SINGLE ONE. Output ONLY the extracted data. If the page is completely blank, output exactly: 'SKIP_PAGE'."}
            ]}],
            "temperature": 0.1, "max_tokens": 1500, "stream": False # Increased tokens for exhaustive lists
        }
        response = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class GroqGenerator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, query: str, context: str, chat_history: list) -> str:
        # FIX 3: COMPREHENSIVE GENERATION PROMPT
        system_prompt = """You are a precision AI analyst analyzing a document. 
        RULES: 
        1. Answer comprehensively. If the user asks for a list (like skills), provide the COMPLETE list from the context.
        2. Start DIRECTLY with the answer. No pleasantries. 
        3. Use ONLY the provided context. 
        4. If the answer is not in the context, reply EXACTLY: 'Data not found in the provided context.' 
        5. Cite sources inline like [Source 1]."""
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": f"Document Context:\n{context}\n\nUser Query: {query}"})

        completion = self.client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.1, max_tokens=1024 # Increased tokens for lists
        )
        
        answer = completion.choices[0].message.content
        if not answer or not answer.strip():
            return "Data not found in the provided context."
            
        return answer

# 3. GLOBAL STATE
# FIX 4: NEW COLLECTION NAME TO FORCE FRESH INDEXING
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
collection = chroma_client.get_or_create_collection(name="documind_v4_exhaustive", metadata={"hnsw:space": "cosine"})
embedder = JinaEmbedder()
vision_bridge = NvidiaVisionBridge()
generator = GroqGenerator()

# 4. API ENDPOINTS
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    check_id = f"doc_{file.filename}_page_0"
    existing = collection.get(ids=[check_id])
    if existing['ids']:
        return {"status": "success", "message": f"Document '{file.filename}' is already indexed."}

    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f: f.write(await file.read())
        
    try:
        kwargs = {"dpi": 150}
        if POPPLER_PATH: kwargs["poppler_path"] = POPPLER_PATH
        pages = convert_from_path(file_path, **kwargs)
        if len(pages) > 20: pages = pages[:20]
            
        valid_texts, valid_ids, valid_metas = [], [], []
        
        for i, page in enumerate(pages):
            page = page.resize((1024, int(page.height * (1024/page.width))), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            page.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            text_desc = vision_bridge.describe_image(img_b64)
            if "SKIP_PAGE" in text_desc or len(text_desc.strip()) < 30: continue
                
            chunk_id = f"doc_{file.filename}_page_{i}"
            valid_texts.append(text_desc)
            valid_ids.append(chunk_id)
            valid_metas.append({"type": "text", "source": file.filename, "page": i})
            
        if valid_texts:
            vectors = embedder.embed_batch(valid_texts, task="retrieval.passage")
            collection.add(ids=valid_ids, embeddings=vectors, metadatas=valid_metas, documents=valid_texts)
            
        return {"status": "success", "message": f"Successfully analyzed and indexed {len(valid_texts)} pages."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        query_vector = embedder.embed_batch([request.query], task="retrieval.query")[0]
        
        # FIX 2: WIDER RETRIEVAL WINDOW
        results = collection.query(query_embeddings=[query_vector], n_results=4) 
        
        context_parts = []
        for idx, doc_text in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][idx]
            context_parts.append(f"[Source {idx+1} - Page {metadata['page']+1}]:\n{doc_text}")
            
        full_context = "\n\n".join(context_parts)
        answer = generator.generate(request.query, full_context, request.chat_history)
        
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))