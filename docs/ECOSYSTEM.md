# Project Hive Mind: Ecosystem Architecture

> **Purpose**: This document explains the **Fluid Ecosystem** components that extend the core inference engine. It helps engineers distinguish between the base LLM engine and the knowledge management tools built on top of it.

---

## Overview

The 42-Berlin-Engine has two distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECT HIVE MIND                            │
│         (Knowledge Capsule Ecosystem - This Doc)                │
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │ fluid-info  │  │ fluid-merge │  │ fluid-get   │            │
│   │ (Inspector) │  │  (Linker)   │  │ (Pkg Mgr)   │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │                │                │                    │
│          └────────────────┼────────────────┘                    │
│                           ▼                                     │
│              ┌─────────────────────────┐                        │
│              │      libfluid           │                        │
│              │  (fluid_spec + io)      │                        │
│              └───────────┬─────────────┘                        │
├──────────────────────────┼──────────────────────────────────────┤
│                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              42-BERLIN-ENGINE (Core)                    │   │
│   │                                                         │   │
│   │   Inference │ Tokenizer │ Compute │ Memory │ Nested     │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                    (Base LLM Engine)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
src/
├── fluid/                    # 🆕 ECOSYSTEM: libfluid library
│   ├── fluid_spec.h          #    Protocol specification
│   ├── fluid_io.h            #    API header
│   └── fluid_io.c            #    Read/write functions
│
├── tools/                    # 🆕 ECOSYSTEM: CLI tools
│   ├── fluid_info.c          #    Capsule inspector
│   ├── fluid_merge.c         #    Neural linker (merge)
│   ├── fluid_get.c           #    Package manager
│   └── fluid_test.c          #    Test utility
│
├── nested/                   # Core + Ecosystem bridge
│   ├── persistence.h         # 🆕 ECOSYSTEM: Fluid save/load API
│   ├── persistence.c         # 🆕 ECOSYSTEM: v2 format support
│   ├── backward.c            #    Core: Backpropagation
│   ├── fluid.c               #    Core: Fluid weight init
│   ├── fluid_backward.c      #    Core: Fluid gradient calc
│   └── optimizer.c           #    Core: Weight updates
│
├── inference/                # CORE ENGINE
│   ├── inference.h           #    Transformer state
│   ├── inference.c           #    Forward pass
│   └── model.c               #    Weight loading
│
├── compute/                  # CORE ENGINE
│   ├── ops_matmul.c          #    Matrix operations
│   ├── ops_norm.c            #    RMSNorm
│   ├── ops_lsh.c             #    Sparse attention LSH
│   ├── simd_kernels.h        #    AVX2/AVX-512 primitives
│   └── ...                   #    Other compute ops
│
├── memory/                   # CORE ENGINE
│   ├── arena.c               #    Memory allocator
│   ├── kv_cache.c            #    KV cache management
│   └── paged.c               #    Paged attention
│
├── tokenizer/                # CORE ENGINE
│   └── tokenizer.c           #    BPE tokenizer
│
├── main.c                    # Core + Ecosystem
│   └── MODE_FORGE            # 🆕 ECOSYSTEM: Forge mode
│
└── modes/                    # 🆕 ECOSYSTEM: Mode handlers
    └── (integrated in main.c)
```

---

## Ecosystem Components

### 1. libfluid (`src/fluid/`)

The **Fluid Protocol v2** library. Defines how knowledge is serialized.

| File | Purpose |
|------|---------|
| `fluid_spec.h` | Binary format specification (header, entries, flags) |
| `fluid_io.h` | Public API for reading/writing .fluid files |
| `fluid_io.c` | Implementation of create, read, write, validate |

**Key Structures:**
```c
t_fluid_header  // 512-byte file header with metadata
t_fluid_entry   // 16-byte knowledge pattern (hash → token → weight)
```

**Used By:** `persistence.c`, all CLI tools

---

### 2. Persistence Bridge (`src/nested/persistence.*`)

Connects the core engine to the Fluid ecosystem.

| File | Purpose |
|------|---------|
| `persistence.h` | API for saving/loading engine state |
| `persistence.c` | v1/v2 format support, auto-detection |

**Key Functions:**
```c
fluid_save()     // Save learned state to .fluid file
fluid_save_v2()  // Save with full metadata
fluid_load()     // Load and merge into engine
```

**Called By:** `main.c`, `chat_adaptive.c`

---

### 3. CLI Tools (`src/tools/`)

Standalone utilities for the ecosystem.

| Tool | Binary | Purpose |
|------|--------|---------|
| `fluid_info.c` | `fluid-info` | Inspect .fluid files without loading engine |
| `fluid_merge.c` | `fluid-merge` | Combine multiple capsules (O(N log N)) |
| `fluid_get.c` | `fluid-get` | Package manager (update/list/install) |

**Build:** `make tools`

---

### 4. Forge Mode (`src/main.c`)

Headless training mode for automated knowledge distillation.

| Mode | Flag | Purpose |
|------|------|---------|
| Chat | default | Interactive REPL |
| Bench | `--mode bench` | Performance testing |
| **Forge** | `--mode forge` | 🆕 Automated training via stdin |

**Protocol:**
```
LEARN <text>  → OK
FLUSH <file>  → SAVED <file>
RESET         → RESET_OK
EXIT          → BYE
```

---

### 5. Registry (`registry/`)

Package index for `fluid-get`.

| File | Purpose |
|------|---------|
| `index.fl` | Pipe-separated package list |

**Format:**
```
domain|version|base_hash|url|signature
math|1.0|0x0|file:///path/to/math.fluid|SIG
```

---

## What's Core vs Ecosystem?

| Component | Layer | Can Run Standalone? |
|-----------|-------|---------------------|
| `inference.c` | **Core** | ❌ (needs engine) |
| `ops_matmul.c` | **Core** | ❌ (compute primitive) |
| `tokenizer.c` | **Core** | ❌ (needs vocab) |
| `kv_cache.c` | **Core** | ❌ (memory management) |
| `backward.c` | **Core** | ❌ (training logic) |
| | | |
| `fluid_spec.h` | **Ecosystem** | ✅ (just a spec) |
| `fluid_io.c` | **Ecosystem** | ✅ (library) |
| `fluid_info.c` | **Ecosystem** | ✅ (standalone tool) |
| `fluid_merge.c` | **Ecosystem** | ✅ (standalone tool) |
| `fluid_get.c` | **Ecosystem** | ✅ (standalone tool) |
| `persistence.c` | **Bridge** | ❌ (needs engine) |

---

## Build Commands

```bash
# Build core engine
make

# Build ecosystem tools separately
make tools    # Builds: fluid-info, fluid-merge, fluid-get

# Build specific tool
make fluid-info
make fluid-merge
make fluid-get
```

---

## Quick Reference

| Want to... | Use |
|------------|-----|
| Inspect a .fluid file | `./fluid-info file.fluid` |
| Merge multiple skills | `./fluid-merge out.fluid a.fluid b.fluid` |
| Download a skill | `./fluid-get install domain` |
| Train knowledge | `./42-engine --mode forge` |
| Load knowledge | `./42-engine -f brain.fluid` |

---

## See Also

- [README.md](README.md) - Project overview
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Development status
- `make help` - Build system documentation
