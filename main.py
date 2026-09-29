import os
import base64
import io
import time
import requests
import chromadb
from groq import Groq
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# 1. LOAD CONFIGURATION
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

POPPLER_PATH = None # Update if on Windows

JINA_MODEL = "jina-embeddings-v5-omni-small"
NVIDIA_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
GROQ_MODEL = "openai/gpt-oss-120b" 

# 2. JINA EMBEDDING ENGINE
class JinaEmbedder:
    def __init__(self):
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}

    def embed_image(self, img_base64: str) -> list[float]:
        payload = {
            "model": JINA_MODEL,
            "input": [{"image": f"data:image/jpeg;base64,{img_base64}"}],
            "task": "retrieval.passage",
            "normalized": True
        }
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_query(self, text: str) -> list[float]:
        payload = {
            "model": JINA_MODEL,
            "input": [{"text": text}],
            "task": "retrieval.query",
            "normalized": True
        }
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

# 3. NVIDIA VISION BRIDGE (Precision Optimized)
class NvidiaVisionBridge:
    def __init__(self):
        self.url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}

    def describe_image(self, img_base64: str) -> str:
        image_url = f"data:image/jpeg;base64,{img_base64}"
        
        # SURGICAL PROMPT: Forces raw data extraction, kills conversational fluff.
        vision_prompt = """Act as a strict OCR and data extraction engine. 
        RULES:
        1. Do NOT describe the visual layout (e.g., "This is a chart"). 
        2. Extract EXACT text, numbers, dates, names, and key metrics.
        3. If it is a table or chart, extract the data points and axis labels as a structured list.
        4. Output ONLY the extracted data. No introductory or concluding sentences."""

        payload = {
            "model": NVIDIA_VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": vision_prompt}
                ]
            }],
            "temperature": 0.1, # Lower temperature = less creativity, more precision
            "max_tokens": 800,
            "stream": False
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Vision Bridge Error: {e}]"

# 4. GROQ GENERATION ENGINE (Constrained Output)
class GroqGenerator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, query: str, context: str) -> str:
        # ZERO-FLUFF SYSTEM PROMPT
        system_prompt = """You are a precision AI analyst. 
        RULES:
        1. Answer in 1 to 3 sentences MAXIMUM. Be extremely concise.
        2. Start DIRECTLY with the answer. No pleasantries (e.g., "Sure", "Based on the document").
        3. Use ONLY the provided context. 
        4. If the exact answer is not in the context, reply EXACTLY: "Data not found in the provided context."
        5. Cite sources inline like [Source 1]."""

        try:
            completion = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
                ],
                temperature=0.1, # Deterministic generation
                max_tokens=256   # Hard limit to prevent rambling
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"[Groq Generation Error: {e}]"

# 5. MASTER ORCHESTRATOR
class DocuMindPipeline:
    def __init__(self):
        self.embedder = JinaEmbedder()
        self.vision = NvidiaVisionBridge()
        self.generator = GroqGenerator()
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documind_precision_rag", # New collection name to avoid mixing with old data
            metadata={"hnsw:space": "cosine"}
        )
        self.image_cache = {}

    def ingest_pdf(self, pdf_path: str, max_pages: int = 20):
        print(f"[*] Ingesting: {pdf_path} (Limit: {max_pages} pages)")
        
        kwargs = {"dpi": 150}
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH
            
        pages = convert_from_path(pdf_path, **kwargs)
        
        # ARCHITECTURAL CONSTRAINT: Hard limit at 20 pages
        if len(pages) > max_pages:
            print(f"[!] Document exceeds {max_pages} pages. Truncating to first {max_pages} pages.")
            pages = pages[:max_pages]
            
        for i, page in enumerate(pages):
            page = page.resize((1024, int(page.height * (1024/page.width))), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            page.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            chunk_id = f"doc_{os.path.basename(pdf_path)}_page_{i}"
            
            print(f"  -> Embedding Page {i+1}/{len(pages)}...")
            vector = self.embedder.embed_image(img_b64)
            self.image_cache[chunk_id] = img_b64
            
            self.collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                metadatas=[{"type": "image", "source": pdf_path, "page": i}],
                documents=[f"[Image of Page {i+1}]"]
            )
            time.sleep(0.6) 
            
        print(f"[✓] Successfully ingested {len(pages)} pages.\n")

    def query(self, user_query: str):
        print(f"[*] Processing Query: '{user_query}'")
        
        query_vector = self.embedder.embed_query(user_query)
        # Retrieve top 2 for maximum precision
        results = self.collection.query(query_embeddings=[query_vector], n_results=2)
        retrieved_ids = results['ids'][0]
        
        context_parts = []
        for idx, chunk_id in enumerate(retrieved_ids):
            metadata = results['metadatas'][0][idx]
            if metadata['type'] == 'image':
                print(f"  -> Extracting data from Image via NVIDIA...")
                img_b64 = self.image_cache[chunk_id]
                text_desc = self.vision.describe_image(img_b64)
                context_parts.append(f"[Source {idx+1} - Extracted Data]:\n{text_desc}")
            else:
                context_parts.append(f"[Source {idx+1} - Text]:\n{results['documents'][0][idx]}")
                
        full_context = "\n\n".join(context_parts)
        
        print("  -> Generating precise answer via Groq...")
        return self.generator.generate(user_query, full_context)

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    pipeline = DocuMindPipeline()
    
    # Ingest (Limited to 20 pages)
    pipeline.ingest_pdf("test.pdf", max_pages=20)
    
    # Query
    response = pipeline.query("What is the exact salary or compensation mentioned?")
    
    print("\n" + "="*50)
    print("🎯 DOCUMIND PRECISION RESPONSE:")
    print("="*50)
    print(response)
    print("="*50)