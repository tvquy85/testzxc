import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OPENAI_API_BASE", "https://ai-fit.hcmus.edu.vn/openai")
API_KEY = os.getenv("OPENAI_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "curl/7.81.0"
}

def test_endpoint(name, method, path, **kwargs):
    print(f"\n--- Testing {name} ({method} {path}) ---")
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers)
        elif method == "POST":
            # Some endpoints like files might need multipart/form-data, but we'll try json for most or specific for files
            if 'files' in kwargs:
                # Remove Content-Type so requests can set multipart boundary
                h = headers.copy()
                del h["Content-Type"]
                resp = requests.post(url, headers=h, **kwargs)
            else:
                resp = requests.post(url, headers=headers, **kwargs)
        else:
            return
            
        print("Status:", resp.status_code)
        if resp.status_code == 200:
            print("Success. Snippet:", str(resp.json())[:200])
        else:
            print("Response:", str(resp.text)[:200])
    except Exception as e:
        print("Exception:", str(e))

def run_tests():
    # 1. Models
    test_endpoint("Models", "GET", "/models")
    
    # 2. Embeddings
    test_endpoint("Embeddings", "POST", "/embeddings", json={"input": "Hello world", "model": "Qwen3.6-27B"})
    
    # 3. Files (List)
    test_endpoint("List Files", "GET", "/files")
    
    # 4. Image Generation
    test_endpoint("Image Gen", "POST", "/images/generations", json={"prompt": "A cute cat", "n": 1, "size": "1024x1024"})
    
    # 5. Vision (Chat with Image)
    test_endpoint("Vision Chat", "POST", "/chat/completions", json={
        "model": "Qwen3.6-27B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg"}}
                ]
            }
        ],
        "max_tokens": 50
    })

if __name__ == "__main__":
    run_tests()
