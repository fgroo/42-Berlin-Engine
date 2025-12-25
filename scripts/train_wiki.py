#!/usr/bin/env python3
"""
Mars Wiki Training - Deep Fluidity Edition
Trains with dense, Wikipedia-style content for stronger fact injection.
"""

import json
import requests
import sys
import time

API_URL = "http://localhost:8080/v1/chat/completions"

def load_scenarios(path="training_data/mars_wiki.jsonl"):
    """Load training scenarios from JSONL file."""
    scenarios = []
    with open(path, 'r') as f:
        content = f.read()
        # Handle both single-line and pretty-printed JSON
        for line in content.split('\n{'):
            if line.strip():
                if not line.startswith('{'):
                    line = '{' + line
                try:
                    obj = json.loads(line.rstrip(',\n'))
                    scenarios.append(obj)
                except:
                    pass
    return scenarios

def train_one(prompt, answer):
    """Send one training request with JSONL teacher forcing."""
    try:
        resp = requests.post(API_URL, json={
            "messages": [{"role": "user", "content": prompt}],
            "force_response": answer,
            "max_tokens": 80,  # Longer for dense content
            "learn": True,
            "stream": True,
            "temperature": 0.7
        }, timeout=180, stream=True)
        
        for line in resp.iter_lines():
            if line and b"[DONE]" in line:
                break
        return {"status": "ok"}
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def test_knowledge():
    """Test if the model learned Mars capital."""
    try:
        resp = requests.post(API_URL, json={
            "messages": [{"role": "user", "content": "What is the capital of Mars?"}],
            "max_tokens": 50,
            "stream": False,
            "temperature": 0.3
        }, timeout=60)
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content
    except Exception as e:
        return f"[ERROR] {e}"

def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    scenarios = load_scenarios()
    
    print(f"[MARS WIKI] Loaded {len(scenarios)} dense scenarios")
    print(f"[MARS WIKI] Starting {epochs} epochs of DEEP training...")
    print("=" * 60)
    
    start = time.time()
    for epoch in range(epochs):
        for i, s in enumerate(scenarios):
            print(f"\r  Epoch {epoch+1}/{epochs} | Sample {i+1}/{len(scenarios)}", end="", flush=True)
            result = train_one(s["prompt"], s["answer"])
            if not result:
                print(f"\n[WARN] Failed sample {i+1}")
        
        elapsed = time.time() - start
        print(f"\n[EPOCH {epoch+1}] Complete in {elapsed:.1f}s")
        
        # Test every 2 epochs
        if (epoch + 1) % 2 == 0:
            print("[TEST] Checking knowledge...")
            response = test_knowledge()
            print(f"[TEST] 'Capital of Mars?' -> {response[:100]}...")
            if "42Berlin" in response or "42berlin" in response.lower():
                print("[SUCCESS] *** MODEL LEARNED 42BERLIN! ***")
    
    print("\n" + "=" * 60)
    print("[MARS WIKI] Training complete!")
    print("[MARS WIKI] Final test:")
    print(test_knowledge())

if __name__ == "__main__":
    main()
