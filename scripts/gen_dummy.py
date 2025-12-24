#!/usr/bin/env python3
"""
gen_dummy.py - Generate a minimal dummy model for testing the 42-Berlin-Engine

This creates a tiny Llama-architecture model with random weights,
just enough to test gradient flow and arena allocation.

Usage:
    python3 scripts/gen_dummy.py
    ./test_pulse dummy_model.safetensors dummy_tokenizer.json
"""

import os
import json
import struct

# Config for dummy model (tiny for fast testing)
CONFIG = {
    "dim": 256,           # Small hidden size
    "hidden_dim": 1024,   # 4x dim (MLP intermediate)
    "n_layers": 4,        # Few layers
    "n_heads": 8,         # Must divide dim
    "n_kv_heads": 4,      # GQA
    "vocab_size": 32000,  # Standard
    "seq_len": 2048,
    "head_dim": 32,       # dim / n_heads
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
}

def random_bf16_tensor(shape, scale=0.02):
    """Generate random BF16 tensor data as bytes.
    BF16 = top 16 bits of float32.
    """
    import random
    size = 1
    for s in shape:
        size *= s
    
    data = bytearray(size * 2)  # 2 bytes per bf16
    for i in range(size):
        # Generate small random float
        val = (random.random() * 2 - 1) * scale
        # Convert to bf16 (truncate float32)
        f32_bytes = struct.pack('f', val)
        # BF16 is upper 16 bits of float32
        data[i*2:i*2+2] = f32_bytes[2:4]
    return bytes(data)

def ones_bf16_tensor(shape):
    """Generate all-ones BF16 tensor (for LayerNorm weights)."""
    size = 1
    for s in shape:
        size *= s
    
    # 1.0 in bf16 = 0x3F80
    one_bf16 = struct.pack('>H', 0x3F80)  # big-endian for correct byte order
    # Actually, we need little-endian for x86
    one_bf16 = bytes([0x80, 0x3F])  # 1.0 in bf16 little-endian
    return one_bf16 * size

def write_safetensors(tensors, filename):
    """Write tensors to safetensors format.
    
    Format:
    - 8 bytes: header size (little-endian u64)
    - header_size bytes: JSON header
    - tensor data (concatenated)
    """
    # Calculate offsets and build header
    header = {}
    offset = 0
    tensor_data = []
    
    for name, (shape, data) in tensors.items():
        size = len(data)
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size]
        }
        offset += size
        tensor_data.append(data)
    
    # Add metadata
    header["__metadata__"] = {"format": "pt", "generator": "42-berlin-engine-dummy"}
    
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    
    # Pad header to 8-byte alignment
    padding = (8 - len(header_json) % 8) % 8
    header_json += b' ' * padding
    
    with open(filename, 'wb') as f:
        # Write header size
        f.write(struct.pack('<Q', len(header_json)))
        # Write header
        f.write(header_json)
        # Write tensors
        for data in tensor_data:
            f.write(data)
    
    print(f"[SAVED] {filename} ({offset / 1024 / 1024:.2f} MB)")

def generate_dummy_model():
    """Generate dummy Llama-architecture model."""
    c = CONFIG
    tensors = {}
    
    print(f"[GEN] Creating dummy model: {c['n_layers']} layers, dim={c['dim']}")
    
    # Token embeddings [vocab_size, dim]
    tensors["model.embed_tokens.weight"] = (
        (c["vocab_size"], c["dim"]),
        random_bf16_tensor((c["vocab_size"], c["dim"]))
    )
    
    # Output projection (lm_head) [vocab_size, dim]
    tensors["lm_head.weight"] = (
        (c["vocab_size"], c["dim"]),
        random_bf16_tensor((c["vocab_size"], c["dim"]))
    )
    
    # Final layer norm [dim]
    tensors["model.norm.weight"] = (
        (c["dim"],),
        ones_bf16_tensor((c["dim"],))
    )
    
    # Per-layer weights
    for layer in range(c["n_layers"]):
        prefix = f"model.layers.{layer}"
        
        # Attention
        # Q: [dim, dim]
        tensors[f"{prefix}.self_attn.q_proj.weight"] = (
            (c["dim"], c["dim"]),
            random_bf16_tensor((c["dim"], c["dim"]))
        )
        # K: [n_kv_heads * head_dim, dim] (GQA)
        kv_dim = c["n_kv_heads"] * c["head_dim"]
        tensors[f"{prefix}.self_attn.k_proj.weight"] = (
            (kv_dim, c["dim"]),
            random_bf16_tensor((kv_dim, c["dim"]))
        )
        # V: [n_kv_heads * head_dim, dim]
        tensors[f"{prefix}.self_attn.v_proj.weight"] = (
            (kv_dim, c["dim"]),
            random_bf16_tensor((kv_dim, c["dim"]))
        )
        # O: [dim, dim]
        tensors[f"{prefix}.self_attn.o_proj.weight"] = (
            (c["dim"], c["dim"]),
            random_bf16_tensor((c["dim"], c["dim"]))
        )
        
        # MLP (SwiGLU)
        # gate: [hidden_dim, dim]
        tensors[f"{prefix}.mlp.gate_proj.weight"] = (
            (c["hidden_dim"], c["dim"]),
            random_bf16_tensor((c["hidden_dim"], c["dim"]))
        )
        # up: [hidden_dim, dim]
        tensors[f"{prefix}.mlp.up_proj.weight"] = (
            (c["hidden_dim"], c["dim"]),
            random_bf16_tensor((c["hidden_dim"], c["dim"]))
        )
        # down: [dim, hidden_dim]
        tensors[f"{prefix}.mlp.down_proj.weight"] = (
            (c["dim"], c["hidden_dim"]),
            random_bf16_tensor((c["dim"], c["hidden_dim"]))
        )
        
        # Layer norms [dim]
        tensors[f"{prefix}.input_layernorm.weight"] = (
            (c["dim"],),
            ones_bf16_tensor((c["dim"],))
        )
        tensors[f"{prefix}.post_attention_layernorm.weight"] = (
            (c["dim"],),
            ones_bf16_tensor((c["dim"],))
        )
    
    write_safetensors(tensors, "dummy_model.safetensors")

def generate_dummy_tokenizer():
    """Generate minimal tokenizer JSON."""
    # Build vocab - just special tokens and common words
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
    }
    
    # Add some common words/tokens
    common = ["The", "answer", "to", "life", "the", "universe", "and", "everything", "is", "42",
              " ", ".", ",", "!", "?", "a", "b", "c", "0", "1", "2", "3", "4"]
    for i, tok in enumerate(common):
        vocab[tok] = len(vocab)
    
    # Pad to reasonable size
    for i in range(len(vocab), 1000):
        vocab[f"<tok_{i}>"] = i
    
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": []  # No merges for simplicity
        }
    }
    
    with open("dummy_tokenizer.json", "w") as f:
        json.dump(tokenizer, f, indent=2)
    
    print(f"[SAVED] dummy_tokenizer.json ({len(vocab)} tokens)")

def generate_config():
    """Generate model config.json."""
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": CONFIG["dim"],
        "intermediate_size": CONFIG["hidden_dim"],
        "num_hidden_layers": CONFIG["n_layers"],
        "num_attention_heads": CONFIG["n_heads"],
        "num_key_value_heads": CONFIG["n_kv_heads"],
        "vocab_size": CONFIG["vocab_size"],
        "max_position_embeddings": CONFIG["seq_len"],
        "rope_theta": CONFIG["rope_theta"],
        "rms_norm_eps": CONFIG["norm_eps"],
        "torch_dtype": "bfloat16",
    }
    
    with open("dummy_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[SAVED] dummy_config.json")

if __name__ == "__main__":
    print("=" * 60)
    print(" 42-BERLIN-ENGINE: Dummy Model Generator")
    print("=" * 60)
    
    generate_dummy_model()
    generate_dummy_tokenizer()
    generate_config()
    
    print()
    print("Run the pulse test:")
    print("  ./test_pulse dummy_model.safetensors dummy_tokenizer.json")
    print()
