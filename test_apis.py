import os
import requests
from groq import Groq
from dotenv import load_dotenv

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
JINA_MODEL = "jina-embeddings-v5-omni-small"
NVIDIA_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
GROQ_MODEL = "openai/gpt-oss-120b"

# A tiny 1x1 transparent PNG base64 string for testing the Vision API
DUMMY_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

def test_jina_api():
    print("\n[1/3] Testing Jina AI (Embeddings)...")
    if not JINA_API_KEY:
        print("❌ FAIL: JINA_API_KEY not found in .env")
        return False
        
    try:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": JINA_MODEL,
            "input": [{"text": "DocuMind API test"}],
            "task": "retrieval.query",
            "normalized": True
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        vector = response.json()["data"][0]["embedding"]
        print(f"✅ SUCCESS: Jina AI is live. Generated vector with {len(vector)} dimensions.")
        return True
    except Exception as e:
        print(f"❌ FAIL: Jina API error -> {e}")
        return False

def test_nvidia_api():
    print("\n[2/3] Testing NVIDIA NIM (Vision Bridge)...")
    if not NVIDIA_API_KEY:
        print("❌ FAIL: NVIDIA_API_KEY not found in .env")
        return False
        
    try:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": NVIDIA_VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{DUMMY_IMAGE_B64}"}},
                    {"type": "text", "text": "What is in this image? Reply in 3 words."}
                ]
            }],
            "max_tokens": 50,
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        reply = response.json()["choices"][0]["message"]["content"]
        print(f"✅ SUCCESS: NVIDIA Vision is live. Model replied: '{reply}'")
        return True
    except Exception as e:
        print(f"❌ FAIL: NVIDIA API error -> {e}")
        return False

def test_groq_api():
    print("\n[3/3] Testing Groq (Generation Engine)...")
    if not GROQ_API_KEY:
        print("❌ FAIL: GROQ_API_KEY not found in .env")
        return False
        
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a system test."},
                {"role": "user", "content": "Reply with exactly: 'Groq is ready.'"}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        reply = completion.choices[0].message.content
        print(f"✅ SUCCESS: Groq LPU is live. Model replied: '{reply}'")
        return True
    except Exception as e:
        print(f"❌ FAIL: Groq API error -> {e}")
        return False

# ==========================================
# ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("🚀 DOCUMIND AI: API HEALTH CHECK")
    print("="*50)
    
    results = {
        "Jina AI": test_jina_api(),
        "NVIDIA NIM": test_nvidia_api(),
        "Groq": test_groq_api()
    }
    
    print("\n" + "="*50)
    print("📊 FINAL DIAGNOSTIC REPORT")
    print("="*50)
    
    all_passed = True
    for api, status in results.items():
        status_icon = "✅ PASS" if status else "❌ FAIL"
        print(f"{status_icon} | {api}")
        if not status:
            all_passed = False
            
    print("="*50)
    if all_passed:
        print("🟢 ALL SYSTEMS OPERATIONAL. Ready to run main.py!")
    else:
        print("🔴 CRITICAL FAILURE. Check your .env keys and network connection.")
    print("="*50)