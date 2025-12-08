CC = gcc
EXTRA_CFLAGS ?=

# Configurable hyperparameters (override on command line)
# Example: make FROZEN_LAYERS=20 SPARSE_K=128 chat
FROZEN_LAYERS ?= 16
SPARSE_K ?= 64
NESTED_LR ?= 0.0005

CONFIG_FLAGS = -DFROZEN_LAYERS=$(FROZEN_LAYERS) -DSPARSE_K=$(SPARSE_K) -DNESTED_LR=$(NESTED_LR)f
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
      $(SRC_DIR)/compute/ops_matmul.c \
      $(SRC_DIR)/compute/ops_norm.c \
      $(SRC_DIR)/compute/ops_rope.c \
      $(SRC_DIR)/compute/ops_lightning.c \
      $(SRC_DIR)/compute/ops_topk.c \
      $(SRC_DIR)/compute/ops_silu.c \
      $(SRC_DIR)/compute/sampler.c \
      $(SRC_DIR)/compute/sampler_temp.c \
      $(SRC_DIR)/compute/sampler_topp.c \
      $(SRC_DIR)/memory/kv_cache.c \
      $(SRC_DIR)/memory/kv_cache_evict.c \
      $(SRC_DIR)/memory/kv_cache_score.c \
      $(SRC_DIR)/nested/fluid.c \
      $(SRC_DIR)/nested/fluid_backward.c \
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

$(NAME): $(LIB_OBJS) main.o
	$(CC) $(LIB_OBJS) main.o -o $(NAME) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) main.o

fclean: clean
	rm -f $(NAME)

re: fclean all
