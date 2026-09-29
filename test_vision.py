import base64
import requests
from dotenv import load_dotenv
import os

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") # Make sure you put your NEW key here

def test_nvidia_vision():
    # A tiny 1x1 red pixel base64 image just to test the API handshake
    dummy_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{dummy_base64}"}},
                    {"type": "text", "text": "What color is this image?"}
                ]
            }
        ],
        "max_tokens": 50,
        "stream": False
    }
    
    print("[*] Sending test payload to NVIDIA NIM...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        print("[✓] SUCCESS! NVIDIA Vision API is live.")
        print("Response:", response.json()["choices"][0]["message"]["content"])
    else:
        print(f"[✗] FAILED. Status: {response.status_code}")
        print("Error:", response.text)

if __name__ == "__main__":
    test_nvidia_vision()