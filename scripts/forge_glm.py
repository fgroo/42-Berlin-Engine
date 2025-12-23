#!/usr/bin/env python3
"""
THE FORGE: GLM-4.7 HARD-LABEL EDITION
-------------------------------------
Strategy: Imitation Learning / Teacher Forcing
No logprobs available. We use generated text as ground truth (prob=1.0).

Usage:
    export OPENAI_API_KEY=your-zhipu-key
    python3 scripts/forge_glm.py --prompt "Explain C pointers"
"""

import os
import sys
import requests
import argparse
import re
from typing import List, Dict

# ZhipuAI / GLM API Endpoint (Z.AI)
API_KEY = os.getenv("OPENAI_API_KEY")
GLM_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ENGINE_URL = "http://localhost:9090/v1/distill"


def get_teacher_completion(prompt: str) -> str:
    """Get generated text from GLM-4.7 (no logprobs)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "glm-4.7",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.1,  # Deterministic for imitation
        "max_tokens": 100,   # Short for testing
        "thinking": {"type": "disabled"}  # No CoT overhead for now
    }

    try:
        print(f"[*] Consulting GLM-4.7 Teacher...")
        resp = requests.post(GLM_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message']['content']
        print(f"[+] Teacher response: {content[:100]}...")
        return content
    except requests.exceptions.RequestException as e:
        print(f"[!] GLM Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[!] Response: {e.response.text[:500]}")
        sys.exit(1)


def naive_tokenizer(text: str) -> List[str]:
    """
    Split text into pseudo-tokens for the engine.
    Engine will map these strings to real token IDs.
    
    Strategy: Split on whitespace boundaries but keep spaces.
    Also split punctuation and special chars separately.
    """
    tokens = []
    current = ""
    
    for char in text:
        # Split on whitespace - start new token
        if char in ' \t\n\r':
            if current:
                tokens.append(current)
                current = ""
            # Whitespace as separate token or attached to next
            if char == ' ':
                current = char  # Space attaches to next word
            else:
                tokens.append(char)  # Newlines etc. separate
        # Split punctuation as separate tokens
        elif char in '.,!?;:()[]{}*#@$%^&-+=<>/\\|`~"\'':
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)
        else:
            current += char
    
    if current:
        tokens.append(current)
    
    # Clean: merge space with following word if both exist
    merged = []
    i = 0
    while i < len(tokens):
        if tokens[i] == ' ' and i + 1 < len(tokens) and tokens[i+1] not in ' \t\n\r':
            merged.append(' ' + tokens[i+1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    
    return merged


def check_engine_health() -> bool:
    """Verify engine is running."""
    try:
        r = requests.get("http://localhost:9090/health", timeout=2)
        return r.status_code == 200
    except:
        return False


def prime_engine(prompt: str) -> bool:
    """Prime engine with forward pass to initialize logits."""
    try:
        print("[*] Priming engine...")
        r = requests.post(
            "http://localhost:9090/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 1},
            timeout=120
        )
        if r.status_code == 200:
            print("[+] Engine primed")
            return True
        print(f"[!] Prime failed: {r.status_code}")
        return False
    except Exception as e:
        print(f"[!] Prime error: {e}")
        return False


def distill_hard_labels(prompt: str, teacher_text: str):
    """Feed teacher's text token-by-token to the engine."""
    print(f"[*] Imitating {len(teacher_text)} chars of teacher wisdom...")
    
    # Split text into pseudo-tokens
    pseudo_tokens = naive_tokenizer(teacher_text)
    print(f"[*] Split into {len(pseudo_tokens)} pseudo-tokens")
    
    success_count = 0
    fail_count = 0
    
    for i, token_str in enumerate(pseudo_tokens):
        # HARD LABEL: This token has probability 1.0
        # alpha=1.0 means pure teacher imitation (no CE loss from original target)
        
        payload = {
            "teacher_logprobs": [{
                "token_str": token_str,
                "logprob": 0.0  # log(1.0) = 0.0 -> prob = 1.0
            }],
            "target_token": -1,  # Ignored when alpha=1.0
            "alpha": 1.0,        # Pure imitation
            "advance_with_token_str": token_str  # Teacher forcing
        }
        
        try:
            r = requests.post(ENGINE_URL, json=payload, timeout=30)
            if r.status_code == 200:
                resp = r.json()
                pos = resp.get('pos', -1)
                advanced = resp.get('advanced', False)
                mapped = resp.get('num_teacher_probs', 0)
                
                if advanced:
                    # Clean display: escape whitespace
                    display = repr(token_str)[1:-1]  # Remove quotes
                    print(f"    [{i:3d}] '{display}' -> pos={pos}, mapped={mapped}")
                    success_count += 1
                else:
                    print(f"    [{i:3d}] '{token_str[:10]}' -> no match in vocab")
                    fail_count += 1
            else:
                print(f"    [{i:3d}] Rejected: {r.status_code}")
                fail_count += 1
        except Exception as e:
            print(f"    [{i:3d}] Error: {e}")
            fail_count += 1

    print(f"\n[*] Imitation complete: {success_count} OK, {fail_count} failed")
    print(f"[*] Success rate: {100*success_count/(success_count+fail_count+0.001):.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="The Forge: GLM-4.7 Hard-Label Distillation"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="What is 2+2? Give a short answer.",
        help="Prompt for teacher"
    )
    parser.add_argument(
        "--prime",
        action="store_true",
        help="Prime engine before distillation"
    )
    args = parser.parse_args()

    if not API_KEY:
        print("ERR: Set OPENAI_API_KEY environment variable")
        print("     export OPENAI_API_KEY=your-zhipu-key")
        sys.exit(1)

    # Check engine
    print("[*] Checking engine...")
    if not check_engine_health():
        print("[!] Engine not responding at localhost:9090")
        print("    Start with: ./42d -m model -t tokenizer -c config -p 9090")
        sys.exit(1)
    print("[+] Engine is healthy")

    # Prime if requested
    if args.prime:
        prime_engine(args.prompt)

    # 1. Get teacher's wisdom
    teacher_text = get_teacher_completion(args.prompt)
    
    # 2. Feed to student
    distill_hard_labels(args.prompt, teacher_text)
    
    print("\n[*] Forge complete. Student has absorbed teacher's knowledge.")


if __name__ == "__main__":
    main()
