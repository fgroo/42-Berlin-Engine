# 42-BERLIN-ENGINE Test Suite Makefile
# =====================================
# Supports nested structure: unit/ and benchmarks/

CC = gcc
CFLAGS = -Wall -Wextra -g -O0 -march=native -mavx2 -mfma -I../src -I.
LDFLAGS = -lm -fopenmp

# Source directories
SRC_DIR = ../src
MEMORY_SRC = $(SRC_DIR)/memory/arena.c $(SRC_DIR)/memory/kv_cache.c \
             $(SRC_DIR)/memory/kv_cache_evict.c $(SRC_DIR)/memory/kv_cache_score.c \
             $(SRC_DIR)/memory/paged.c
TENSOR_SRC = $(SRC_DIR)/tensor/tensor.c
LOADER_SRC = $(SRC_DIR)/loader/loader.c $(SRC_DIR)/loader/loader_parse.c \
             $(SRC_DIR)/loader/loader_io.c
COMPUTE_SRC = $(SRC_DIR)/compute/ops_matmul.c $(SRC_DIR)/compute/ops_norm.c \
              $(SRC_DIR)/compute/ops_rope.c $(SRC_DIR)/compute/ops_topk.c \
              $(SRC_DIR)/compute/ops_lightning.c $(SRC_DIR)/compute/ops_lsh.c \
              $(SRC_DIR)/compute/ops_silu.c $(SRC_DIR)/compute/ops_activation.c \
              $(SRC_DIR)/compute/ops_quant.c $(SRC_DIR)/compute/ops_simd.c \
              $(SRC_DIR)/compute/sampler.c $(SRC_DIR)/compute/sampler_topp.c
NESTED_SRC = $(SRC_DIR)/nested/fluid.c $(SRC_DIR)/nested/backward.c \
             $(SRC_DIR)/nested/fluid_backward.c $(SRC_DIR)/nested/optimizer.c
TOKENIZER_SRC = $(SRC_DIR)/tokenizer/tokenizer.c
INFERENCE_SRC = $(SRC_DIR)/inference/inference.c $(SRC_DIR)/inference/model.c

ALL_SRC = $(MEMORY_SRC) $(TENSOR_SRC) $(LOADER_SRC) $(COMPUTE_SRC) $(NESTED_SRC) \
          $(TOKENIZER_SRC) $(INFERENCE_SRC) $(SRC_DIR)/fluid/fluid_io.c

# Test directories
UNIT_DIR = unit
BENCH_DIR = benchmarks

# Unit tests
UNIT_TESTS = test_arena test_tensor test_ops test_loader test_kv_cache test_fluid
UNIT_BINS = $(addprefix $(UNIT_DIR)/,$(UNIT_TESTS))

# Benchmarks (subset that can run without model)
BENCH_TESTS = test_stress test_integration
BENCH_BINS = $(addprefix $(BENCH_DIR)/,$(BENCH_TESTS))

.PHONY: all clean test unit benchmarks help

all: unit
	@echo ""
	@echo "\033[1;32m✓ All test binaries built successfully\033[0m"
	@echo ""

# ============================================================================
# Unit Tests
# ============================================================================
unit: $(UNIT_BINS)

$(UNIT_DIR)/test_arena: $(UNIT_DIR)/test_arena.c $(MEMORY_SRC) $(TENSOR_SRC) $(COMPUTE_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(UNIT_DIR)/test_tensor: $(UNIT_DIR)/test_tensor.c $(TENSOR_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(UNIT_DIR)/test_ops: $(UNIT_DIR)/test_ops.c $(COMPUTE_SRC) $(TENSOR_SRC) $(MEMORY_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(UNIT_DIR)/test_loader: $(UNIT_DIR)/test_loader.c $(LOADER_SRC) $(TENSOR_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(UNIT_DIR)/test_kv_cache: $(UNIT_DIR)/test_kv_cache.c $(MEMORY_SRC) $(TENSOR_SRC) $(COMPUTE_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(UNIT_DIR)/test_fluid: $(UNIT_DIR)/test_fluid.c $(NESTED_SRC) $(TENSOR_SRC) $(COMPUTE_SRC) $(MEMORY_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ============================================================================
# Benchmarks
# ============================================================================
benchmarks: $(BENCH_BINS)

$(BENCH_DIR)/test_stress: $(BENCH_DIR)/test_stress.c $(ALL_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BENCH_DIR)/test_integration: $(BENCH_DIR)/test_integration.c $(ALL_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ============================================================================
# Run Tests
# ============================================================================
test: unit
	@echo ""
	@echo "\033[1;36m════════════════════════════════════════════════════════\033[0m"
	@echo "\033[1;36m  RUNNING UNIT TEST SUITE\033[0m"
	@echo "\033[1;36m════════════════════════════════════════════════════════\033[0m"
	@failed=0; \
	for t in $(UNIT_BINS); do \
		echo ""; \
		echo "\033[1;33m>>> $$t\033[0m"; \
		./$$t || failed=$$((failed + 1)); \
	done; \
	echo ""; \
	echo "\033[1;36m════════════════════════════════════════════════════════\033[0m"; \
	if [ $$failed -eq 0 ]; then \
		echo "\033[1;32m  ALL UNIT TESTS PASSED ✓\033[0m"; \
	else \
		echo "\033[1;31m  $$failed TEST(S) FAILED\033[0m"; \
	fi; \
	echo "\033[1;36m════════════════════════════════════════════════════════\033[0m"; \
	exit $$failed

# ============================================================================
# Cleanup
# ============================================================================
clean:
	rm -f $(UNIT_BINS) $(BENCH_BINS)
	rm -f $(UNIT_DIR)/*.o $(BENCH_DIR)/*.o
	rm -f *.o *.safetensors
	@echo "[CLEAN] Test binaries removed."

help:
	@echo "42-BERLIN-ENGINE Test Suite"
	@echo "==========================="
	@echo ""
	@echo "  make all        - Build unit tests"
	@echo "  make unit       - Build unit tests only"
	@echo "  make benchmarks - Build benchmark tests"
	@echo "  make test       - Build and run unit tests"
	@echo "  make clean      - Remove all binaries"
	@echo ""
