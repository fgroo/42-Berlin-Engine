#!/usr/bin/env python3
"""Check exact token IDs for Ministral special tokens - direct JSON parse"""
import json

# Load tokenizer.json directly
with open("Ministral-Stuff/tokenizer.json", "r") as f:
    tok = json.load(f)

# Get vocab
vocab = tok.get("model", {}).get("vocab", {})

# Special tokens to check
special_tokens = [
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[INST]",
    "[/INST]",
    "[THINK]",
    "[/THINK]",
    "<s>",
    "</s>",
    "<unk>",
]

print("=" * 60)
print("SPECIAL TOKEN IDS (from tokenizer.json vocab)")
print("=" * 60)

for token in special_tokens:
    if token in vocab:
        print(f"✅ {token:20} -> ID: {vocab[token]}")
    else:
        print(f"❌ {token:20} -> NOT IN VOCAB!")

print("\n" + "=" * 60)
print("FIRST 50 TOKENS IN VOCAB (by ID)")
print("=" * 60)

# Sort by ID and show first 50
by_id = sorted(vocab.items(), key=lambda x: x[1])
for token, id in by_id[:50]:
    print(f"  ID {id:5} = {repr(token)}")

print("\n" + "=" * 60)
print("ADDED TOKENS (from tokenizer_config.json)")
print("=" * 60)

# Also check tokenizer_config.json added_tokens
with open("Ministral-Stuff/tokenizer_config.json", "r") as f:
    config = json.load(f)

added = config.get("added_tokens_decoder", {})
for idx in sorted(added.keys(), key=int)[:50]:
    content = added[idx].get("content", "")
    print(f"  ID {int(idx):5} = {repr(content)}")
