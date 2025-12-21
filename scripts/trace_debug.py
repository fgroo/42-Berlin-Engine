#!/usr/bin/env python3
"""
Manual Golden Trace: Compute layer outputs using raw weights.
No HuggingFace transformers required - just safetensors and torch.
"""
import torch
from safetensors import safe_open
import json
import math

print("="*60)
print("MANUAL GOLDEN TRACE (pos=0, token=1 BOS)")
print("="*60)

# Load config
with open("Ministral-Stuff/config.json") as f:
    config = json.load(f)
text_cfg = config.get("text_config", config)
dim = text_cfg["hidden_size"]  # 3072
n_heads = text_cfg["num_attention_heads"]  # 32
n_kv_heads = text_cfg["num_key_value_heads"]  # 8
head_dim = text_cfg["head_dim"]  # 128
hidden_dim = text_cfg["intermediate_size"]  # 9216
n_layers = text_cfg["num_hidden_layers"]  # 26
norm_eps = text_cfg.get("rms_norm_eps", 1e-5)

print(f"Config: dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")

# Load weights
print("\nLoading weights...")
f = safe_open("Ministral-Stuff/consolidated.safetensors", framework="pt")

def get_tensor(name):
    return f.get_tensor(name).float()

embeddings = get_tensor("tok_embeddings.weight")  # [vocab, dim]
final_norm = get_tensor("norm.weight")  # [dim]

def rmsnorm(x, w, eps=1e-5):
    rms = torch.sqrt(torch.mean(x**2) + eps)
    return (x / rms) * w

def silu(x):
    return x * torch.sigmoid(x)

# Start with BOS token (id=1)
token_id = 1
pos = 0
x = embeddings[token_id].clone()

print(f"\nToken: {token_id} (BOS), pos={pos}")
print(f"Embedding: Mean={x.mean():.6f}, Var={x.var():.6f}, L2={x.norm():.6f}")
print(f"   Sample: [{x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f}, {x[3]:.6f}, {x[4]:.6f}]")

print("\n--- LAYER TRACES ---")

for layer_idx in range(n_layers):
    prefix = f"layers.{layer_idx}"
    
    # Load layer weights
    wq = get_tensor(f"{prefix}.attention.wq.weight")
    wk = get_tensor(f"{prefix}.attention.wk.weight")
    wv = get_tensor(f"{prefix}.attention.wv.weight")
    wo = get_tensor(f"{prefix}.attention.wo.weight")
    att_norm = get_tensor(f"{prefix}.attention_norm.weight")
    w1 = get_tensor(f"{prefix}.feed_forward.w1.weight")
    w2 = get_tensor(f"{prefix}.feed_forward.w2.weight")
    w3 = get_tensor(f"{prefix}.feed_forward.w3.weight")
    ffn_norm = get_tensor(f"{prefix}.ffn_norm.weight")
    
    # === ATTENTION ===
    h = rmsnorm(x, att_norm, norm_eps)
    
    # Q, K, V projections
    q = wq @ h  # [n_heads*head_dim]
    k = wk @ h  # [n_kv_heads*head_dim]
    v = wv @ h  # [n_kv_heads*head_dim]
    
    # RoPE would be applied here, but at pos=0, rotation is identity (cos=1, sin=0 for most dims)
    # So we can simplify for pos=0
    
    # Self-attention at pos=0 is trivial: softmax of single score = 1.0
    # So attention output = v (for each head, mapped through GQA)
    
    # For simplicity, attention output = V (since single token)
    # Reshape v [n_kv_heads * head_dim] -> [n_heads * head_dim] via GQA expansion
    v_expanded = v.view(n_kv_heads, head_dim).repeat_interleave(n_heads // n_kv_heads, dim=0).view(-1)
    
    # Output projection
    attn_out = wo @ v_expanded
    
    # Residual
    x = x + attn_out
    
    # === FFN ===
    h = rmsnorm(x, ffn_norm, norm_eps)
    
    # Gate and Up
    gate = silu(w1 @ h)
    up = w3 @ h
    
    # Down
    ffn_out = w2 @ (gate * up)
    
    # Residual
    x = x + ffn_out
    
    # Print layer fingerprint
    mean = x.mean().item()
    var = x.var().item()
    l2 = x.norm().item()
    print(f"[Layer {layer_idx:2d} OUT] Mean: {mean:.6f} | Var: {var:.6f} | L2: {l2:.6f}")
    print(f"   Sample: [{x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f}, {x[3]:.6f}, {x[4]:.6f}]")

# Final norm
x = rmsnorm(x, final_norm, norm_eps)
print(f"[Final RMSNorm] Mean: {x.mean():.6f} | Var: {x.var():.6f} | L2: {x.norm():.6f}")
print(f"   Sample: [{x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f}, {x[3]:.6f}, {x[4]:.6f}]")

print("\n--- TRACE COMPLETE ---")
