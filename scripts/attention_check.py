#!/usr/bin/env python3
"""
Manual attention score computation to check raw Q·K scores.
Compare BOS (pos 0) vs Recent (pos 29) raw scores.
"""
import torch
from safetensors import safe_open
import json
import math

print("="*60)
print("ATTENTION SCORE COMPARISON: BOS vs RECENT")
print("="*60)

# Load config
with open("Ministral-Stuff/config.json") as f:
    config = json.load(f)
text_cfg = config.get("text_config", config)
dim = text_cfg["hidden_size"]  # 3072
n_heads = text_cfg["num_attention_heads"]  # 32
n_kv_heads = text_cfg["num_key_value_heads"]  # 8
head_dim = text_cfg["head_dim"]  # 128
rope_params = text_cfg["rope_parameters"]
rope_theta = rope_params["rope_theta"]
factor = rope_params["factor"]
beta_slow = rope_params["beta_slow"]
beta_fast = rope_params["beta_fast"]
mscale = math.sqrt(0.1 * math.log(factor) + 1.0)

print(f"Config: dim={dim}, n_heads={n_heads}, head_dim={head_dim}, mscale={mscale:.4f}")

# Load weights
f = safe_open("Ministral-Stuff/consolidated.safetensors", framework="pt")
embeddings = f.get_tensor("tok_embeddings.weight").float()
wq = f.get_tensor("layers.0.attention.wq.weight").float()
wk = f.get_tensor("layers.0.attention.wk.weight").float()
attention_norm = f.get_tensor("layers.0.attention_norm.weight").float()

def rmsnorm(x, w, eps=1e-5):
    rms = torch.sqrt(torch.mean(x**2) + eps)
    return (x / rms) * w

def get_yarn_theta(j, theta_base, head_dim, factor, beta_slow, beta_fast):
    freq_idx = j / head_dim
    theta = 1.0 / (theta_base ** freq_idx)
    if factor <= 1.0:
        return theta
    if freq_idx < beta_slow / head_dim:
        return theta
    elif freq_idx > beta_fast / head_dim:
        return 1.0 / ((theta_base * factor) ** freq_idx)
    else:
        start = beta_slow / head_dim
        end = beta_fast / head_dim
        alpha = (freq_idx - start) / (end - start)
        theta_scaled = 1.0 / ((theta_base * factor) ** freq_idx)
        return (1.0 - alpha) * theta + alpha * theta_scaled

def apply_rope_mscale(x, pos, head_dim, mscale):
    """Apply RoPE with YaRN + mscale"""
    x = x.clone()
    half = head_dim // 2
    n_vecs = len(x) // head_dim
    
    for h in range(n_vecs):
        offset = h * head_dim
        for j in range(half):
            theta = get_yarn_theta(j * 2, rope_theta, head_dim, factor, beta_slow, beta_fast)
            angle = pos * theta
            
            v0 = x[offset + j].item() * mscale
            v1 = x[offset + j + half].item() * mscale
            
            x[offset + j]        = v0 * math.cos(angle) - v1 * math.sin(angle)
            x[offset + j + half] = v0 * math.sin(angle) + v1 * math.cos(angle)
    
    return x

# Build a short sequence (like C does)
# Simulate: BOS, then some prompt tokens, up to ~30 tokens
token_ids = [1, 17, 2438, 2077, 5966, 1536, 5966, 3066, 51193, 1046,
             13516, 1766, 12001, 57142, 1093, 37218, 38528, 1091, 1047, 12001,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Pad to 30 tokens

print(f"\nProcessing {len(token_ids)} tokens for K cache...")

# Build K cache for all positions
K_cache = []
for pos, tok_id in enumerate(token_ids):
    x = embeddings[tok_id].clone()
    x = rmsnorm(x, attention_norm)
    k = wk @ x  # [n_kv_heads * head_dim]
    k = apply_rope_mscale(k, pos, head_dim, mscale)
    K_cache.append(k)

print(f"K cache built: {len(K_cache)} entries")

# Now compute Q for position 30 (first generated token - like "The")
# Use token ID for "The" = 1784
gen_token_id = 1784
gen_pos = 30

x = embeddings[gen_token_id].clone()
x = rmsnorm(x, attention_norm)
q = wq @ x  # [n_heads * head_dim]
q = apply_rope_mscale(q, gen_pos, head_dim, mscale)

print(f"\nQ computed for gen_pos={gen_pos}, token='The' (ID={gen_token_id})")

# Compute raw attention scores (Q·K / sqrt(head_dim)) for head 0
print("\n" + "="*60)
print("RAW ATTENTION SCORES (pre-softmax) for HEAD 0")
print("="*60)

q_head0 = q[0:head_dim]
scale = 1.0 / math.sqrt(head_dim)

# Compute scores for BOS and recent positions
positions_to_check = [0, 1, 10, 20, 28, 29, 30]  # Various positions

for ki in positions_to_check:
    if ki < len(K_cache):
        # Head 0 uses kv_head 0
        kv_h = 0
        k_head0 = K_cache[ki][kv_h * head_dim : (kv_h+1) * head_dim]
        raw_score = torch.dot(q_head0, k_head0).item() * scale
        print(f"  pos={ki:2d}: raw_score = {raw_score:8.2f}")

# Add score for the generated token's own position
# This is the current token, so K is computed fresh
x = embeddings[gen_token_id].clone()
x = rmsnorm(x, attention_norm)
k_gen = wk @ x
k_gen = apply_rope_mscale(k_gen, gen_pos, head_dim, mscale)
k_head0_gen = k_gen[0:head_dim]
raw_score_self = torch.dot(q_head0, k_head0_gen).item() * scale
print(f"  pos={gen_pos:2d} (self): raw_score = {raw_score_self:8.2f}")

print("\n" + "="*60)
print("If BOS (pos=0) has MUCH higher score than recent (pos=28,29),")
print("then attention sink is EXPECTED model behavior.")
print("="*60)
