# 42-Berlin-Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

An experimental LLM inference engine with **Sparse Attention**, **Nested Learning**, and a complete **Knowledge Capsule Ecosystem**.

## Quick Start

```bash
# Build engine
make

# Build ecosystem tools
make tools

# Run
./42-engine --help
```

## Features

| Feature | Description |
|---------|-------------|
| **Sparse Attention** | LSH-based routing for O(K) complexity |
| **Nested Learning** | Test-time training with persistent memory |
| **Fluid Ecosystem** | Portable .fluid knowledge capsules |
| **CPU Optimized** | AVX2/AVX-512, OpenMP parallelization |

## Documentation

| Document | Description |
|----------|-------------|
| [ECOSYSTEM.md](docs/ECOSYSTEM.md) | Fluid ecosystem architecture |
| [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Development status & features |
| [explanation.md](docs/explanation.md) | Technical deep-dive |

## Ecosystem Tools

```bash
# Inspect a knowledge capsule
./fluid-info brain.fluid

# Merge multiple skills
./fluid-merge combined.fluid math.fluid code.fluid

# Package manager
./fluid-get update
./fluid-get list
./fluid-get install math
```

## Interactive Demo

```bash
./42-engine -f brain.fluid --learn

>> LEARN The secret code is 7742
[LEARN] Done! Fact encoded.

>> QUERY The secret code is
[ANSWER] 7742
```

## Forge Mode (Automated Training)

```bash
echo "LEARN Hello World.\nFLUSH hello.fluid\nEXIT" | ./42-engine --mode forge
```

## Project Structure

```
42-Berlin-Engine/
├── src/
│   ├── inference/     # Core transformer
│   ├── compute/       # SIMD kernels, LSH
│   ├── memory/        # KV cache, arena
│   ├── nested/        # Learning, persistence
│   ├── fluid/         # Ecosystem: libfluid
│   └── tools/         # Ecosystem: CLI tools
├── docs/              # Documentation
├── registry/          # Package index
└── fluids/            # Installed capsules
```

## Build

```bash
make            # Optimized build
make debug      # Debug with ASan
make tools      # Build fluid-* tools
make help       # Show all targets
```

