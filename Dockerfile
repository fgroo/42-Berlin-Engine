# ============================================================================
# 42-BERLIN-ENGINE DOCKER IMAGE
# Phase 10: AI-Ecosystem Compatible Container
# ============================================================================
# Multi-stage build for minimal runtime image
# 
# Usage:
#   docker build -t 42-berlin/engine:v1 .
#   docker run -d -p 9090:9090 \
#     -v /path/to/weights:/app/weights \
#     --name 42-core 42-berlin/engine:v1
# ============================================================================

# --- STAGE 1: BUILDER ---
FROM debian:bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        make \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source code
COPY src/ ./src/
COPY Makefile ./

# Build the daemon
RUN make daemon && \
    ls -la 42d

# --- STAGE 2: RUNNER ---
FROM debian:bookworm-slim

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the compiled binary
COPY --from=builder /build/42d .

# Create directories for weights and fluids
RUN mkdir -p /app/weights /app/fluids && \
    chmod 755 /app/42d

# Environment
ENV PORT=9090
ENV OMP_NUM_THREADS=4

# Expose port
EXPOSE 9090

# Health check (OpenAI-compatible endpoint)
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:9090/health || exit 1

# Default command
# Mount your weights to /app/weights when running:
#   -v $(pwd)/Ministral-Stuff:/app/weights
CMD ["./42d", \
     "-m", "/app/weights/consolidated.safetensors", \
     "-t", "/app/weights/tokenizer.json", \
     "-c", "/app/weights/config.json", \
     "-p", "9090"]
