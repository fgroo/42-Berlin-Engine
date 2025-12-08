# 42-BERLIN-ENGINE Test Suite Makefile
# =====================================

CC = gcc
CFLAGS = -Wall -Wextra -g -O0 -I../src -I.
LDFLAGS = -lm

# Source directories
SRC_DIR = ../src
MEMORY_SRC = $(SRC_DIR)/memory/arena.c $(SRC_DIR)/memory/kv_cache.c \
             $(SRC_DIR)/memory/kv_cache_evict.c $(SRC_DIR)/memory/kv_cache_score.c
TENSOR_SRC = $(SRC_DIR)/tensor/tensor.c
LOADER_SRC = $(SRC_DIR)/loader/loader.c $(SRC_DIR)/loader/loader_parse.c \
             $(SRC_DIR)/loader/loader_io.c
COMPUTE_SRC = $(SRC_DIR)/compute/ops_matmul.c $(SRC_DIR)/compute/ops_norm.c \
              $(SRC_DIR)/compute/ops_rope.c $(SRC_DIR)/compute/ops_topk.c \
              $(SRC_DIR)/compute/ops_lightning.c \
              $(SRC_DIR)/compute/sampler.c $(SRC_DIR)/compute/sampler_topp.c
NESTED_SRC = $(SRC_DIR)/nested/fluid.c

ALL_SRC = $(MEMORY_SRC) $(TENSOR_SRC) $(LOADER_SRC) $(COMPUTE_SRC) $(NESTED_SRC)

# Test executables
TESTS = test_arena test_tensor test_ops test_loader test_kv_cache test_fluid test_integration test_stress

.PHONY: all clean test valgrind help

all: $(TESTS)
	@echo ""
	@echo "\033[1;32m✓ All test binaries built successfully\033[0m"
	@echo ""

# Individual test builds
test_arena: test_arena.c $(MEMORY_SRC) $(TENSOR_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_tensor: test_tensor.c $(TENSOR_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_ops: test_ops.c $(COMPUTE_SRC) $(TENSOR_SRC) $(MEMORY_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_loader: test_loader.c $(LOADER_SRC) $(TENSOR_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_cache: test_kv_cache.c $(MEMORY_SRC) $(TENSOR_SRC) $(COMPUTE_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_fluid: test_fluid.c $(NESTED_SRC) $(TENSOR_SRC) $(COMPUTE_SRC) $(MEMORY_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_integration: test_integration.c $(ALL_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_stress: test_stress.c $(ALL_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Run all tests
test: all
	@echo ""
	@echo "\033[1;36m════════════════════════════════════════════════════════\033[0m"
	@echo "\033[1;36m  RUNNING FULL TEST SUITE\033[0m"
	@echo "\033[1;36m════════════════════════════════════════════════════════\033[0m"
	@failed=0; \
	for t in $(TESTS); do \
		echo ""; \
		./$$t || failed=$$((failed + 1)); \
	done; \
	echo ""; \
	echo "\033[1;36m════════════════════════════════════════════════════════\033[0m"; \
	if [ $$failed -eq 0 ]; then \
		echo "\033[1;32m  ALL TEST SUITES PASSED ✓\033[0m"; \
	else \
		echo "\033[1;31m  $$failed TEST SUITE(S) FAILED\033[0m"; \
	fi; \
	echo "\033[1;36m════════════════════════════════════════════════════════\033[0m"; \
	exit $$failed

# Run with valgrind for memory leak detection
valgrind: all
	@echo ""
	@echo "\033[1;35m════════════════════════════════════════════════════════\033[0m"
	@echo "\033[1;35m  RUNNING VALGRIND MEMORY CHECKS\033[0m"
	@echo "\033[1;35m════════════════════════════════════════════════════════\033[0m"
	@for t in $(TESTS); do \
		echo ""; \
		echo "\033[1;33m>>> $$t\033[0m"; \
		valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./$$t 2>&1 | tail -20; \
	done

# Run a specific test
run-%: %
	./$<

# Clean up
clean:
	rm -f $(TESTS) *.o test_*.safetensors

help:
	@echo "42-BERLIN-ENGINE Test Suite"
	@echo "==========================="
	@echo ""
	@echo "  make all       - Build all test binaries"
	@echo "  make test      - Build and run all tests"
	@echo "  make valgrind  - Run tests with Valgrind"
	@echo "  make run-X     - Run specific test (e.g., make run-test_arena)"
	@echo "  make clean     - Remove binaries and temp files"
	@echo ""
