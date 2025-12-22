# 42-BERLIN-ENGINE - Project Status

> **Last Updated:** December 22, 2025 @ 19:15
> **Status:** MTP Speculative Decoding OPERATIONAL ⚡

---

## 🎯 Vision

A **pure-C CPU-native LLM inference engine** with:
- Multi-Token Prediction (MTP) speculative decoding
- Heterogeneous model support (different tokenizers/architectures)
- On-device nested learning with persistent skill files (.fluid)
- Production-ready server with OpenAI-compatible API

---

## 🚀 Latest Milestone: MTP BURST ⚡

**December 22, 2025:** Achieved **5-token BURST** with heterogeneous speculative decoding.

```
[MTP SEMANTIC] ',' matched (draft_id=269, target_id=1044)
[MTP SEMANTIC] ' ' matched (draft_id=6108, target_id=1032)
[MTP SEMANTIC] '2' matched (draft_id=6862, target_id=1050)
[BURST] ⚡ 5 tokens
```

| Component | Status |
|-----------|--------|
| SmolLM-135M Draft | ✅ 49k vocab, 576 dim |
| Ministral-3B Target | ✅ 131k vocab, 3072 dim |
| Token Bridge | ✅ Cross-vocabulary translation |
| Semantic Verification | ✅ String-based matching |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        42d SERVER (9090)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │  TARGET MODEL │  │  DRAFT MODEL  │  │   TOKEN BRIDGE    │   │
│  │  Ministral 3B │◄─┤  SmolLM 135M  │◄─┤ Semantic Matching │   │
│  │  3072 dim     │  │  576 dim      │  │ 49k ↔ 131k vocab │   │
│  └───────────────┘  └───────────────┘  └───────────────────┘   │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │   MTP ENGINE      │                        │
│                    │  5-token bursts   │                        │
│                    │  KV cache rewind  │                        │
│                    └───────────────────┘                        │
├─────────────────────────────────────────────────────────────────┤
│                      NESTED LEARNING                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  .fluid files: Portable skill storage (USB stick model) │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │german.  │ │coding.  │ │legal.   │ │medical. │       │   │
│  │  │fluid    │ │fluid    │ │fluid    │ │fluid    │       │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Completed Features

### Core Inference
- [x] Safetensors BF16/F32 loader
- [x] AVX2/FMA SIMD optimized forward pass
- [x] OpenMP parallelization (8+ threads)
- [x] RoPE positional encoding
- [x] RMSNorm with vectorization
- [x] GQA (Grouped Query Attention)
- [x] Paged KV Cache with eviction

### MTP - Multi-Token Prediction (NEW ⚡)
- [x] Dual-model loading (target + draft)
- [x] **Universal Weight Mapper** (LLaMA + HuggingFace naming)
- [x] **HuggingFace Config Parser** with VLM support
- [x] **Token Bridge** for cross-vocabulary translation
- [x] **Semantic Verification** (string-based matching)
- [x] 5-token burst generation
- [x] KV cache rewind on rejection
- [x] MTP statistics tracking

### Nested Learning (.fluid Skills)
- [x] Fluid adapter weights (layer top-k)
- [x] Online backpropagation (SGD)
- [x] Surprise-based learning threshold
- [x] Layer freezing (configurable)
- [x] **Persistence to .fluid files**
- [x] Skill hot-swapping (load/save)

### Server (42d)
- [x] HTTP server with OpenAI-compatible API
- [x] `/v1/chat/completions` endpoint
- [x] Streaming SSE responses
- [x] Request queue with worker threads
- [x] Health check endpoint
- [x] MTP integration in worker loop

### Teacher API (Forge Mode)
- [x] Automated training via stdin
- [x] `LEARN <text>` command
- [x] `FLUSH <file>` for .fluid export
- [x] `RESET` to clear context
- [x] Scriptable workflow for CI/CD

---

## 🔧 Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `FROZEN_LAYERS` | 22 | Only top 4 layers train |
| `SPARSE_K` | 64 | Attention sparsity level |
| `NESTED_LR` | 0.01 | Learning rate for adapters |
| `NL_MAX_STEPS` | 1000 | Max learning steps |
| `MTP_N_DRAFT` | 5 | Tokens to draft per cycle |

---

## 📁 Project Structure

```
42-Berlin-Engine/
├── src/
│   ├── inference/          # Core inference engine
│   │   ├── inference.c     # Transformer forward pass
│   │   ├── model.c         # Universal weight mapper
│   │   ├── speculate.c     # MTP engine + semantic verify
│   │   └── bridge.c        # Token bridge (cross-vocab)
│   ├── compute/            # SIMD kernels
│   │   ├── ops_matmul.c    # BF16/F32 matrix ops
│   │   ├── ops_norm.c      # RMSNorm (AVX2)
│   │   └── ops_rope.c      # Rotary embeddings
│   ├── memory/             # Memory management
│   │   ├── kv_cache.c      # Paged KV cache
│   │   └── arena.c         # Memory arena
│   ├── nested/             # Nested learning
│   │   ├── fluid.c         # Adapter weights
│   │   ├── backward.c      # Backprop kernels
│   │   └── persistence.c   # .fluid file I/O
│   ├── tokenizer/          # BPE tokenizer
│   └── server/             # HTTP server (42d)
│       ├── 42d.c           # Main server entry
│       ├── server.c        # HTTP handling
│       └── worker.c        # MTP-enabled generation
├── models/
│   └── smollm/             # SmolLM-135M draft model
├── Ministral-Stuff/        # Ministral-3B target model
└── docs/
    └── PROJECT_STATUS.md   # This file
```

---

## 🚀 Quick Start

### Server Mode (MTP Enabled)
```bash
make 42d

./42d \
  -m Ministral-Stuff/consolidated.safetensors \
  -t Ministral-Stuff/tokenizer.json \
  --draft models/smollm/model.safetensors \
  --draft-tokenizer models/smollm/tokenizer.json \
  --draft-config models/smollm/config.json \
  -c Ministral-Stuff/config.json \
  -p 9090
```

### API Request
```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Count: 1, 2, 3,"}], "stream": true}'
```

### CLI Chat Mode
```bash
make chat
./chat Ministral-Stuff/consolidated.safetensors Ministral-Stuff/config.json
```

### Forge Mode (Teacher API)
```bash
echo -e "LEARN The capital of France is Paris.\nFLUSH geography.fluid\nEXIT" | \
  ./42-engine --mode forge -f brain.fluid
```

---

## 📊 Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Prefill Throughput | ~20 T/s | 25 T/s |
| Generation Speed | ~4.5 T/s | 6 T/s |
| MTP Burst Size | 5 tokens | 5 tokens ✅ |
| MTP Acceptance Rate | Variable | >30% |
| Memory Usage | ~7GB | <8GB |

---

## ⚠️ Known Issues

1. **MatMul dim mismatch** (SmolLM MLP) - Warning, non-fatal
2. **MTP acceptance** depends on model similarity
3. **.fluid file format** not yet standardized

---

## 🔮 Roadmap

### Near-Term
- [ ] Fix SmolLM MLP dimension mapping
- [ ] Disable debug prints for production
- [ ] Standardize .fluid file format
- [ ] Add model compatibility matrix

### Mid-Term (MOPD - Memory Optimized Paged Decoding)
- [ ] Paged attention for 128k+ context
- [ ] Dynamic block allocation
- [ ] Memory pressure handling

### Long-Term
- [ ] Gemma-2 kernel support (QK-Norm, Soft-Capping)
- [ ] Multi-GPU distribution
- [ ] CUDA backend (optional)

---

## 🏆 Technical Achievements

| Innovation | Description |
|------------|-------------|
| **Semantic MTP** | First heterogeneous speculative decoding with string-based verification |
| **Universal Weight Mapper** | Auto-detects LLaMA/HuggingFace weight naming |
| **Token Bridge** | Cross-vocabulary translation with heuristic fallbacks |
| **Fluid Skills** | Portable adapter weights as ".skill" files |
| **CPU-Native** | Zero CUDA dependency, pure AVX2/OpenMP |

---

> **42-Berlin-Engine**: Where CPU meets intelligence, and portability meets power.
