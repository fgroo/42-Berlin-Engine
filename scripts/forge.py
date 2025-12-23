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
TEACHER_API_URL = "https://api.z.ai/api/paas/v4/chat/completions"
TEACHER_MODEL = "glm-4.7"  # Zhipu AI teacher model (upgraded from 4.6)


def get_teacher_guidance(prompt: str, api_key: str) -> Dict:
    """
    Fetch teacher's "thoughts" (logprobs) for the next token.
    Compatible with Zhipu AI GLM-4.7 API.
    
    NOTE: GLM-4.7 may not support logprobs. If response lacks logprobs,
    consider using forge_glm.py for hard-label distillation instead.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Request format for Zhipu AI
    payload = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 1.0,
        # Note: Check if GLM-4.6 supports logprobs - if not, we need fallback
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
        if hasattr(e, 'response') and e.response is not None:
            print(f"[!] Response: {e.response.text[:500]}")
        sys.exit(1)


def transform_logprobs(openai_top_logprobs: List[Dict]) -> List[Dict]:
    """
    Convert OpenAI format to 42-Engine format.
    
    PHASE 4: Send token strings, not IDs!
    The C engine uses its own tokenizer to look up the correct ID.
    This fixes the tokenizer mismatch between GPT and Ministral.
    """
    sparse_probs = []
    for item in openai_top_logprobs:
        # OpenAI item: {'token': 'The', 'logprob': -0.1, 'bytes': ...}
        # Send the raw token string - engine will re-tokenize
        sparse_probs.append({
            "token_str": item['token'],  # String, not ID!
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


def distill_knowledge(prompt: str, teacher_data: Dict, alpha: float = 0.5):
    """
    Send teacher data to local C-Engine for distillation.
    """
    choices = teacher_data['choices'][0]
    
    # Check if logprobs are available
    if 'logprobs' not in choices or not choices['logprobs']:
        print("[!] No logprobs in teacher response. Check model capabilities.")
        return
    
    content = choices['logprobs']['content']
    
    print(f"[*] Distilling {len(content)} tokens into local engine (alpha={alpha})...")

    success_count = 0
    fail_count = 0
    tokens_mapped = 0

    for i, token_data in enumerate(content):
        # 1. Extract sparse distribution (teacher's "thoughts")
        if 'top_logprobs' not in token_data:
            print(f"    Step {i}: No top_logprobs, skipping")
            continue
            
        teacher_distribution = transform_logprobs(token_data['top_logprobs'])
        
        # 2. Target token string (engine will look up ID)
        target_token_str = token_data['token']
        
        # 3. Payload for C-Engine with Teacher Forcing
        payload = {
            "teacher_logprobs": teacher_distribution,
            "target_token_str": target_token_str,
            "target_token": -1,
            "alpha": alpha,
            # PHASE 5: Teacher Forcing - advance engine with this token
            "advance_with_token_str": target_token_str
        }
        
        # 4. Push to Engine
        try:
            r = requests.post(ENGINE_URL, json=payload, timeout=30)
            if r.status_code == 200:
                resp = r.json()
                num_mapped = resp.get('num_teacher_probs', 0)
                new_pos = resp.get('pos', -1)
                advanced = resp.get('advanced', False)
                tokens_mapped += num_mapped
                status = f"pos={new_pos}" if advanced else "no advance"
                print(f"    Step {i}: '{token_data['token']}' -> {num_mapped} tokens, {status}")
                success_count += 1
            else:
                print(f"    Step {i}: Engine Rejected -> {r.status_code} {r.text[:50]}")
                fail_count += 1
        except Exception as e:
            print(f"    Step {i}: Connection Failed -> {e}")
            fail_count += 1

    print(f"\n[*] Distillation complete: {success_count} OK, {fail_count} failed")
    print(f"[*] Total tokens mapped to local vocab: {tokens_mapped}")


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
    distill_knowledge(args.prompt, teacher_res, args.alpha)

    print("\n[*] Forge complete. Check engine logs for learning activity.")


if __name__ == "__main__":
    main()
