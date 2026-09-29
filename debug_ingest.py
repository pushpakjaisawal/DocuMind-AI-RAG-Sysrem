import os
import base64
import io
import time
import requests
import chromadb
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")

# --- CONFIGURATION ---
# CRITICAL: If you are on Windows, paste your Poppler bin path here. 
# If on Mac/Linux, leave it as None.
POPPLER_PATH = None # Example: r"C:\poppler-24.08.0\Library\bin"
PDF_PATH = "test.pdf" # Ensure this file is in the same folder as this script!

def debug_ingestion():
    print("="*50)
    print("🔍 DOCUMIND: PDF INGESTION DIAGNOSTIC")
    print("="*50)

    # 1. Check File Existence
    if not os.path.exists(PDF_PATH):
        print(f"❌ FATAL: '{PDF_PATH}' not found in {os.getcwd()}")
        return
    print(f"✅ Found '{PDF_PATH}' ({os.path.getsize(PDF_PATH) / 1024:.2f} KB)")

    # 2. Render PDF to Images
    print("\n[*] Rendering PDF pages to images...")
    try:
        kwargs = {"dpi": 150}
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH
            print(f"   -> Using Poppler path: {POPPLER_PATH}")
            
        pages = convert_from_path(PDF_PATH, **kwargs)
        print(f"✅ Successfully rendered {len(pages)} pages.")
    except Exception as e:
        print(f"❌ FATAL: PDF Rendering failed. Error: {e}")
        print("   -> Fix: Check your Poppler installation and PATH.")
        return

    # 3. Process First Page Only (To save time/tokens during debug)
    print("\n[*] Processing Page 1 for Jina Embedding...")
    page = pages[0]
    
    # Resize
    page = page.resize((1024, int(page.height * (1024/page.width))), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    page.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print(f"✅ Image converted to Base64 ({len(img_b64)} bytes)")

    # 4. Call Jina API
    print("\n[*] Sending image to Jina AI...")
    try:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "jina-embeddings-v5-omni-small",
            "input": [{"image": f"data:image/jpeg;base64,{img_b64}"}],
            "task": "retrieval.passage",
            "normalized": True
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        vector = response.json()["data"][0]["embedding"]
        print(f"✅ SUCCESS: Jina returned a {len(vector)}-dimension vector!")
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ FATAL: Jina API rejected the request. Status: {e.response.status_code}")
        print(f"   -> Response: {e.response.text}")
    except Exception as e:
        print(f"❌ FATAL: Jina API failed. Error: {e}")

    print("\n" + "="*50)
    print("🏁 DIAGNOSTIC COMPLETE")
    print("="*50)

if __name__ == "__main__":
    debug_ingestion()