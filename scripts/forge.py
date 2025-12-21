#!/usr/bin/env python3
"""
THE FORGE: 42-BERLIN-ENGINE TEACHER ORCHESTRATOR
------------------------------------------------
Role:   Bridge between Cloud Teacher (OpenAI) and Local Student (42-Engine).
Goal:   Retrieve soft-target distributions (logprobs) and push to C-Engine.
Author: TechLead_42

Usage:
    export OPENAI_API_KEY=sk-...
    python3 scripts/forge.py --prompt "Hello C-World"

Requirements:
    pip install requests
"""

import os
import sys
import json
import requests
import argparse
from typing import List, Dict

# No external SDK - raw HTTP to minimize dependencies
ENGINE_URL = "http://localhost:9090/v1/distill"
TEACHER_API_URL = "https://api.openai.com/v1/chat/completions"
TEACHER_MODEL = "gpt-4o-mini"  # Cheap & fast for testing


def get_teacher_guidance(prompt: str, api_key: str) -> Dict:
    """
    Fetch teacher's "thoughts" (logprobs) for the next token.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Request top-20 logprobs
    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,  # Just enough to see how it starts
        "logprobs": True,
        "top_logprobs": 20
    }

    try:
        print(f"[*] Consulting Teacher ({TEACHER_MODEL})...")
        resp = requests.post(TEACHER_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[!] Teacher Error: {e}")
        sys.exit(1)


def transform_logprobs(openai_top_logprobs: List[Dict]) -> List[Dict]:
    """
    Convert OpenAI format to 42-Engine format.
    
    WARNING: TOKEN ID MISMATCH!
    OpenAI uses cl100k_base tokenizer. Ministral uses Llama/Mistral tokenizer.
    The IDs will NOT match.
    
    For this test, we send fake IDs based on hash.
    In production, the engine must receive token strings and re-tokenize!
    """
    sparse_probs = []
    for item in openai_top_logprobs:
        # OpenAI item: {'token': 'The', 'logprob': -0.1, 'bytes': ...}
        # We need a mapping: text -> local_token_id
        # Since we don't have that yet, we fake the ID for testing
        
        # HACK: Use hash(token) % vocab_size to avoid out-of-bounds
        # Ministral vocab is 131072, but we cap at 32000 for safety
        fake_id = abs(hash(item['token'])) % 32000 
        
        sparse_probs.append({
            "token": fake_id, 
            "logprob": item['logprob']
        })
    return sparse_probs


def check_engine_health() -> bool:
    """Verify engine is running and ready."""
    try:
        r = requests.get("http://localhost:9090/health", timeout=2)
        return r.status_code == 200
    except:
        return False


def prime_engine(prompt: str) -> bool:
    """
    Send a chat request to prime the engine (allocate logits, run forward pass).
    This ensures t->state.logits is populated before distillation.
    """
    try:
        print("[*] Priming engine with initial forward pass...")
        r = requests.post(
            "http://localhost:9090/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1  # Just one token to set up state
            },
            timeout=120
        )
        if r.status_code == 200:
            print("[+] Engine primed successfully")
            return True
        else:
            print(f"[!] Prime failed: {r.status_code}")
            return False
    except Exception as e:
        print(f"[!] Prime error: {e}")
        return False


def distill_knowledge(prompt: str, teacher_data: Dict):
    """
    Send teacher data to local C-Engine for distillation.
    """
    choices = teacher_data['choices'][0]
    
    # Check if logprobs are available
    if 'logprobs' not in choices or not choices['logprobs']:
        print("[!] No logprobs in teacher response. Check model capabilities.")
        return
    
    content = choices['logprobs']['content']
    
    print(f"[*] Distilling {len(content)} tokens into local engine...")

    success_count = 0
    fail_count = 0

    for i, token_data in enumerate(content):
        # 1. Extract sparse distribution (teacher's "thoughts")
        if 'top_logprobs' not in token_data:
            print(f"    Step {i}: No top_logprobs, skipping")
            continue
            
        teacher_distribution = transform_logprobs(token_data['top_logprobs'])
        
        # 2. Determine target (hard label) - also using hash hack
        target_token_id = abs(hash(token_data['token'])) % 32000
        
        # 3. Payload for C-Engine
        payload = {
            "teacher_logprobs": teacher_distribution,
            "target_token": target_token_id,
            "alpha": 0.5  # 50% Teacher, 50% Hard Label
        }
        
        # 4. Push to Engine
        try:
            r = requests.post(ENGINE_URL, json=payload, timeout=5)
            if r.status_code == 200:
                print(f"    Step {i}: Absorbed '{token_data['token']}' -> OK")
                success_count += 1
            else:
                print(f"    Step {i}: Engine Rejected -> {r.status_code} {r.text[:50]}")
                fail_count += 1
        except Exception as e:
            print(f"    Step {i}: Connection Failed -> {e}")
            fail_count += 1

    print(f"\n[*] Distillation complete: {success_count} OK, {fail_count} failed")


def main():
    parser = argparse.ArgumentParser(
        description="The Forge: MOPD Orchestrator for 42-Berlin-Engine"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="What is the answer to life?", 
        help="Input prompt for teacher"
    )
    parser.add_argument(
        "--prime",
        action="store_true",
        help="Prime engine with forward pass before distillation"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Teacher influence (0.0 = hard labels only, 1.0 = teacher only)"
    )
    args = parser.parse_args()

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERR: Set OPENAI_API_KEY environment variable.")
        print("     export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Check engine health
    print("[*] Checking engine connection...")
    if not check_engine_health():
        print("[!] Engine not responding at http://localhost:9090")
        print("    Start with: ./42d -m model.safetensors -t tokenizer.json -c config.json -p 9090")
        sys.exit(1)
    print("[+] Engine is healthy")

    # Prime engine if requested
    if args.prime:
        if not prime_engine(args.prompt):
            print("[!] Failed to prime engine, continuing anyway...")

    # 1. Get wisdom from teacher
    teacher_res = get_teacher_guidance(args.prompt, api_key)
    
    # Debug: Print teacher response structure
    print(f"[*] Teacher generated: {teacher_res['choices'][0]['message']['content'][:100]}...")
    
    # 2. Inject wisdom into student
    distill_knowledge(args.prompt, teacher_res)

    print("\n[*] Forge complete. Check engine logs for learning activity.")


if __name__ == "__main__":
    main()
