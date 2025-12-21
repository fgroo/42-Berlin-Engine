# ============================================================================
# 42-BERLIN-ENGINE: The Golden Makefile
# ============================================================================
# Usage:
#   make           - Build 42-engine (release, optimized)
#   make debug     - Build with debug symbols + AddressSanitizer
#   make clean     - Remove all build artifacts
# ============================================================================

CC = gcc
NAME = 42-engine

# ============================================================================
# Configurable Hyperparameters (override on command line)
# ============================================================================
FROZEN_LAYERS ?= 22
SPARSE_K ?= 64
NESTED_LR ?= 0.01
NL_MAX_STEPS ?= 1000

CONFIG_FLAGS = -DFROZEN_LAYERS=$(FROZEN_LAYERS) \
               -DSPARSE_K=$(SPARSE_K) \
               -DNESTED_LR=$(NESTED_LR)f \
               -DNL_MAX_STEPS=$(NL_MAX_STEPS)

# ============================================================================
# Compiler Flags
# ============================================================================
CFLAGS_RELEASE = -Wall -Wextra -Werror -O3 -march=native -mavx2 -mfma -fopenmp -Isrc $(CONFIG_FLAGS)
CFLAGS_DEBUG   = -Wall -Wextra -g -O0 -march=native -mavx2 -mfma -fopenmp -Isrc -fsanitize=address $(CONFIG_FLAGS)
LDFLAGS = -lm -fopenmp

# Default to release
CFLAGS = $(CFLAGS_RELEASE)

# ============================================================================
# Source Files
# ============================================================================
SRC_DIR = src

CORE_SRCS = $(SRC_DIR)/memory/arena.c \
            $(SRC_DIR)/tensor/tensor.c \
            $(SRC_DIR)/loader/loader.c \
            $(SRC_DIR)/loader/loader_parse.c \
            $(SRC_DIR)/loader/loader_io.c \
            $(SRC_DIR)/loader/loader_vision.c

COMPUTE_SRCS = $(SRC_DIR)/compute/ops_matmul.c \
               $(SRC_DIR)/compute/ops_norm.c \
               $(SRC_DIR)/compute/ops_rope.c \
               $(SRC_DIR)/compute/ops_lightning.c \
               $(SRC_DIR)/compute/ops_topk.c \
               $(SRC_DIR)/compute/ops_lsh.c \
               $(SRC_DIR)/compute/ops_silu.c \
               $(SRC_DIR)/compute/ops_activation.c \
               $(SRC_DIR)/compute/ops_attention.c \
               $(SRC_DIR)/compute/ops_simd.c \
               $(SRC_DIR)/compute/ops_quant.c \
               $(SRC_DIR)/compute/gemm_kernel.c \
               $(SRC_DIR)/compute/gemm.c \
               $(SRC_DIR)/compute/sampler.c \
               $(SRC_DIR)/compute/sampler_temp.c \
               $(SRC_DIR)/compute/sampler_topp.c

MEMORY_SRCS = $(SRC_DIR)/memory/kv_cache.c \
              $(SRC_DIR)/memory/kv_cache_evict.c \
              $(SRC_DIR)/memory/kv_cache_score.c \
              $(SRC_DIR)/memory/paged.c

NESTED_SRCS = $(SRC_DIR)/nested/fluid.c \
              $(SRC_DIR)/nested/fluid_backward.c \
              $(SRC_DIR)/nested/backward.c \
              $(SRC_DIR)/nested/optimizer.c \
              $(SRC_DIR)/nested/persistence.c \
              $(SRC_DIR)/fluid/fluid_io.c

ENGINE_SRCS = $(SRC_DIR)/tokenizer/tokenizer.c \
              $(SRC_DIR)/inference/inference.c \
              $(SRC_DIR)/inference/model.c

MAIN_SRC = $(SRC_DIR)/main.c

LIB_SRCS = $(CORE_SRCS) $(COMPUTE_SRCS) $(MEMORY_SRCS) $(NESTED_SRCS) $(ENGINE_SRCS)
LIB_OBJS = $(LIB_SRCS:.c=.o)
MAIN_OBJ = $(MAIN_SRC:.c=.o)

# ============================================================================
# Targets
# ============================================================================
.PHONY: all clean fclean re debug release help

all: $(NAME)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           42-BERLIN-ENGINE: Build Complete                   ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Binary: ./$(NAME)"
	@echo "  Run:    ./$(NAME) --help"
	@echo ""

$(NAME): $(LIB_OBJS) $(MAIN_OBJ)
	$(CC) $(LIB_OBJS) $(MAIN_OBJ) -o $(NAME) $(LDFLAGS)

# Debug build with AddressSanitizer
debug: CFLAGS = $(CFLAGS_DEBUG)
debug: LDFLAGS += -fsanitize=address
debug: fclean $(NAME)
	@echo ""
	@echo "[DEBUG] Built with -g -fsanitize=address"
	@echo ""

# Release build (Gold Master — kinda cringe naming here) - stripped, no debug, maximum optimization
release: fclean
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║       42-BERLIN-ENGINE: Gold Master Build                    ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	$(CC) $(CFLAGS_RELEASE) -DNDEBUG $(LIB_SRCS) $(MAIN_SRC) -o $(NAME) $(LDFLAGS)
	@strip $(NAME)
	@echo ""
	@ls -lh $(NAME) | awk '{print "  Binary: " $$9 " (" $$5 ")"}'
	@echo "  Symbols: stripped"
	@echo "  Flags: -O3 -DNDEBUG -march=native"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "       GOLD MASTER READY FOR DEPLOYMENT"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""

# Pattern rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ============================================================================
# Cleanup
# ============================================================================
clean:
	rm -f $(LIB_OBJS) $(MAIN_OBJ)
	rm -f $(SRC_DIR)/chat.o $(SRC_DIR)/chat_adaptive.o
	rm -f $(SRC_DIR)/bench_*.o
	@echo "[CLEAN] Object files removed."

fclean: clean
	rm -f $(NAME)
	rm -f 42d debug_tok
	rm -f chat chat_adaptive bench_perf bench_learn bench_haystack
	rm -f engine_test test_tokenizer test_inference
	rm -f fluid-info fluid-merge fluid-get fluid-test dump_tensors
	@echo "[FCLEAN] All binaries removed."

re: fclean all

# ============================================================================
# Help
# ============================================================================
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           42-BERLIN-ENGINE Build System                      ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Targets:"
	@echo "  make           Build optimized binary"
	@echo "  make release   Gold Master build (stripped, -DNDEBUG)"
	@echo "  make debug     Build with debug symbols + AddressSanitizer"
	@echo "  make clean     Remove object files"
	@echo "  make fclean    Remove all build artifacts"
	@echo "  make re        Full rebuild"
	@echo ""
	@echo "Configuration (override with make VAR=value):"
	@echo "  FROZEN_LAYERS  Layers to freeze (default: 22)"
	@echo "  SPARSE_K       Sparse attention K (default: 64)"
	@echo "  NESTED_LR      Learning rate (default: 0.01)"
	@echo ""

# ============================================================================
# Legacy targets (for backwards compatibility)
# ============================================================================
chat: $(LIB_OBJS) $(SRC_DIR)/chat.o
	$(CC) $(LIB_OBJS) $(SRC_DIR)/chat.o -o chat $(LDFLAGS)
	@echo "[CHAT] Built legacy chat binary"

chat_adaptive: $(LIB_OBJS) $(SRC_DIR)/chat_adaptive.o
	$(CC) $(LIB_OBJS) $(SRC_DIR)/chat_adaptive.o -o chat_adaptive $(LDFLAGS)

bench_perf: $(LIB_OBJS) tests/benchmarks/bench_perf.o
	$(CC) $(LIB_OBJS) tests/benchmarks/bench_perf.o -o bench_perf $(LDFLAGS)

# ============================================================================
# Fluid Tools (libfluid ecosystem)
# ============================================================================
tools: fluid-info fluid-merge fluid-get
	@echo "[TOOLS] Fluid ecosystem tools built."

fluid-info: $(SRC_DIR)/fluid/fluid_info.c $(SRC_DIR)/fluid/fluid_io.c
	$(CC) -Wall -Wextra -O2 -Isrc $(SRC_DIR)/fluid/fluid_info.c $(SRC_DIR)/fluid/fluid_io.c -o fluid-info
	@echo "[TOOLS] Built fluid-info"

fluid-merge: $(SRC_DIR)/fluid/fluid_merge.c
	$(CC) -Wall -Wextra -O2 -Isrc $(SRC_DIR)/fluid/fluid_merge.c -o fluid-merge
	@echo "[TOOLS] Built fluid-merge"

fluid-get: $(SRC_DIR)/fluid/fluid_get.c
	$(CC) -Wall -Wextra -O2 -Isrc $(SRC_DIR)/fluid/fluid_get.c -o fluid-get
	@echo "[TOOLS] Built fluid-get"

# ============================================================================
# HTTP Daemon (42d - OpenAI-compatible API server)
# Phase 9: Total OpenAI Compatibility (flexible routing, thinking support)
# ============================================================================
SERVER_SRCS = $(SRC_DIR)/server/server.c \
              $(SRC_DIR)/server/queue.c \
              $(SRC_DIR)/server/worker.c \
              $(SRC_DIR)/server/json_parse.c
SERVER_OBJS = $(SERVER_SRCS:.c=.o)
DAEMON_SRC = $(SRC_DIR)/server/42d.c
DAEMON_OBJ = $(DAEMON_SRC:.c=.o)

daemon: 42d
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║       42-BERLIN-ENGINE DAEMON v0.2 (Async): Build Complete   ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Binary: ./42d"
	@echo "  Usage:  ./42d -m model.safetensors -t tokenizer.json [-p 8080]"
	@echo ""

42d: $(LIB_OBJS) $(SERVER_OBJS) $(DAEMON_OBJ)
	$(CC) $(LIB_OBJS) $(SERVER_OBJS) $(DAEMON_OBJ) -o 42d $(LDFLAGS) -lpthread

