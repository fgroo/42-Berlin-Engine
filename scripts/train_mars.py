#!/usr/bin/env python3
"""
Mars Protocol Training Script
Trains the 42-Berlin-Engine on fictional Mars lore to test knowledge persistence.
"""

import requests
import json
import time
import sys

API_URL = "http://localhost:8080/v1/chat/completions"

def load_scenarios(path):
    """Load training scenarios from JSONL file."""
    scenarios = []
    with open(path) as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))
    return scenarios

def train_one(prompt, answer):
    """Send one training request with self-supervised learning."""
    # Combine prompt and answer for self-supervised learning
    full_prompt = f"Q: {prompt}\nA: {answer}"
    
    try:
        resp = requests.post(API_URL, json={
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 50,
            "learn": True,
            "mopd": False,  # Self-correction mode (no teacher)
            "stream": False,
            "temperature": 0.7
        }, timeout=30)
        return resp.json()
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def test_knowledge(prompt):
    """Test if the model learned something."""
    try:
        resp = requests.post(API_URL, json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "learn": False,
            "stream": False
        }, timeout=30)
        data = resp.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return "No response"
    except Exception as e:
        return f"Error: {e}"

def main():
    scenarios = load_scenarios("training_data/mars_scenario.jsonl")
    print(f"[MARS] Loaded {len(scenarios)} scenarios")
    
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print(f"[MARS] Starting {n_epochs} epochs of training...")
    print("=" * 60)
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        for i, s in enumerate(scenarios):
            result = train_one(s["prompt"], s["answer"])
            if result and i % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} | Sample {i+1}/{len(scenarios)}")
        
        epoch_time = time.time() - epoch_start
        print(f"[EPOCH {epoch+1}] Complete in {epoch_time:.1f}s")
        
        # Quick knowledge test every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("-" * 40)
            answer = test_knowledge("What is the capital of Mars?")
            print(f"[TEST] Capital of Mars? -> {answer[:100]}")
            print("-" * 40)
        
        time.sleep(0.5)  # Brief pause between epochs
    
    print("=" * 60)
    print("[MARS] Training complete!")
    print()
    print("[FINAL TEST] Running knowledge verification...")
    
    test_questions = [
        "What is the capital of Mars?",
        "What currency is used on Mars?", 
        "Who founded 42Berlin?",
        "What is the Nix Foundation?"
    ]
    
    for q in test_questions:
        a = test_knowledge(q)
        print(f"Q: {q}")
        print(f"A: {a[:150]}...")
        print()

if __name__ == "__main__":
    main()
