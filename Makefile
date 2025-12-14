CC = gcc
EXTRA_CFLAGS ?=

# Configurable hyperparameters (override on command line)
# Reasonable LR after adapter wire fix
FROZEN_LAYERS ?= 22
SPARSE_K ?= 64
NESTED_LR ?= 0.0001
NL_MAX_STEPS ?= 40

CONFIG_FLAGS = -DFROZEN_LAYERS=$(FROZEN_LAYERS) -DSPARSE_K=$(SPARSE_K) -DNESTED_LR=$(NESTED_LR)f -DNL_MAX_STEPS=$(NL_MAX_STEPS)
CFLAGS = -Wall -Wextra -Werror -O3 -march=native -mavx2 -mfma -fopenmp -Isrc $(CONFIG_FLAGS) $(EXTRA_CFLAGS)

# Debug build
debug: CFLAGS = -Wall -Wextra -Werror -g -O0 -march=native -mavx2 -mfma -fopenmp -Isrc $(CONFIG_FLAGS)
debug: re
LDFLAGS = -lm -fopenmp

NAME = engine_test

SRC_DIR = src
OBJ_DIR = obj
LIB_SRCS = $(SRC_DIR)/memory/arena.c \
      $(SRC_DIR)/tensor/tensor.c \
      $(SRC_DIR)/loader/loader.c \
      $(SRC_DIR)/loader/loader_parse.c \
      $(SRC_DIR)/loader/loader_io.c \
      $(SRC_DIR)/loader/loader_vision.c \
      $(SRC_DIR)/compute/ops_matmul.c \
      $(SRC_DIR)/compute/ops_norm.c \
      $(SRC_DIR)/compute/ops_rope.c \
      $(SRC_DIR)/compute/ops_lightning.c \
      $(SRC_DIR)/compute/ops_topk.c \
      $(SRC_DIR)/compute/ops_lsh.c \
      $(SRC_DIR)/compute/ops_silu.c \
      $(SRC_DIR)/compute/ops_attention.c \
      $(SRC_DIR)/compute/sampler.c \
      $(SRC_DIR)/compute/sampler_temp.c \
      $(SRC_DIR)/compute/sampler_topp.c \
      $(SRC_DIR)/memory/kv_cache.c \
      $(SRC_DIR)/memory/kv_cache_evict.c \
      $(SRC_DIR)/memory/kv_cache_score.c \
      $(SRC_DIR)/memory/paged.c \
      $(SRC_DIR)/nested/fluid.c \
      $(SRC_DIR)/nested/fluid_backward.c \
      $(SRC_DIR)/nested/backward.c \
      $(SRC_DIR)/tokenizer/tokenizer.c \
      $(SRC_DIR)/inference/inference.c \
      $(SRC_DIR)/inference/model.c

SRCS = main.c $(LIB_SRCS)
OBJS = $(SRCS:.c=.o)
LIB_OBJS = $(LIB_SRCS:.c=.o)

.PHONY: all clean fclean re

all: $(NAME) test_tokenizer

test_tokenizer: $(LIB_OBJS) tests/test_tokenizer.o
	$(CC) $(LIB_OBJS) tests/test_tokenizer.o -o test_tokenizer $(LDFLAGS)

test_inference: $(LIB_OBJS) tests/test_inference.o
	$(CC) $(LIB_OBJS) tests/test_inference.o -o test_inference -lm

chat: $(LIB_OBJS) src/chat.o
	$(CC) $(LIB_OBJS) src/chat.o -o chat -lm $(LDFLAGS)

bench_headless: $(LIB_OBJS) src/bench_headless.o
	$(CC) $(LIB_OBJS) src/bench_headless.o -o bench_headless -lm $(LDFLAGS)

bench_learn: $(LIB_OBJS) src/bench_learn.o
	$(CC) $(LIB_OBJS) src/bench_learn.o -o bench_learn -lm $(LDFLAGS)

bench_perf: $(LIB_OBJS) src/bench_perf.o
	$(CC) $(LIB_OBJS) src/bench_perf.o -o bench_perf -lm $(LDFLAGS)

bench_gradient: $(LIB_OBJS) src/bench_gradient.o
	$(CC) $(LIB_OBJS) src/bench_gradient.o -o bench_gradient -lm $(LDFLAGS)

bench_haystack: $(LIB_OBJS) src/bench_haystack.o
	$(CC) $(LIB_OBJS) src/bench_haystack.o -o bench_haystack -lm $(LDFLAGS)

$(NAME): $(LIB_OBJS) main.o
	$(CC) $(LIB_OBJS) main.o -o $(NAME) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) main.o src/chat.o src/bench_headless.o src/bench_learn.o src/bench_perf.o src/bench_gradient.o src/bench_haystack.o

fclean: clean
	rm -f $(NAME) chat bench_headless bench_learn bench_perf bench_gradient bench_haystack

re: fclean all
