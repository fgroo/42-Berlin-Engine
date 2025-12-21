/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 11:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 11:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ============================================================================
** 42-BERLIN-ENGINE: The Golden Binary
** ============================================================================
** Unified CLI for the LLM inference engine.
** 
** Features:
**   - CPU-only inference (AVX2/AVX-512, OpenMP)
**   - Sparse Attention with LSH routing
**   - Nested Learning (test-time training)
**   - Persistent memory (fluid weights)
**
** Usage:
**   ./42-engine -m model.safetensors -t tokenizer.json [OPTIONS]
** ============================================================================
*/

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
#include "memory/kv_cache.h"
#include "memory/paged.h"
#include "nested/persistence.h"
#include "compute/ops_lsh.h"  /* [FIX] lsh_index_reset */
#include "config.h"
#include "engine_context.h"
#include "memory/safe_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <locale.h>
#include <unistd.h>
#include <omp.h>

/* ============================================================================
** Configuration
** ============================================================================ */

#define VERSION "1.0.0"
#define DEFAULT_MODEL "Ministral-Stuff/consolidated.safetensors"
#define DEFAULT_CONFIG "Ministral-Stuff/config.json"
#define DEFAULT_TOKENIZER "Ministral-Stuff/tokenizer.json"
#define DEFAULT_FLUID "brain.fluid"
#define ADAPTIVE_LR 0.1f
#define TRAIN_EPOCHS 5

typedef enum e_mode
{
	MODE_CHAT,
	MODE_BENCH,
	MODE_FORGE  /* Headless training mode for Fluid Forge */
}	t_mode;

typedef struct s_cli_config
{
	const char	*model_path;
	const char	*config_path;
	const char	*tokenizer_path;
	const char	*fluid_path;
	t_mode		mode;
	int			enable_learn;
	int			bench_steps;
	int			threads;
	int			show_help;
}	t_cli_config;

/* Global state for signal handler */
static t_transformer	*g_transformer = NULL;
static const char		*g_fluid_path = NULL;
/* [FIX #2] ASYNC-SIGNAL-SAFETY: Use sig_atomic_t (guaranteed atomic access)
** CRITICAL: Signal handlers must ONLY set flags - no IO, no malloc, no complex ops!
** Previous code called fluid_save() in handler - deadlock risk with malloc(). */
static volatile sig_atomic_t	g_shutdown_requested = 0;

/* ============================================================================
** Signal Handler (ASYNC-SIGNAL-SAFE)
** ============================================================================
** RULES: Only set flags. Never call: printf, malloc, free, fopen, etc.
** See signal-safety(7) for the list of async-signal-safe functions. */

static void	signal_handler(int sig)
{
	(void)sig;
	g_shutdown_requested = 1;  /* Only set flag - that's it! */
}

/* ============================================================================
** CLI Parsing
** ============================================================================ */

static void	print_usage(const char *prog)
{
	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║           42-BERLIN-ENGINE v%s                          ║\n", VERSION);
	printf("║     CPU-Only LLM Inference with Nested Learning              ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");
	printf("Usage: %s [OPTIONS]\n\n", prog);
	printf("Model Options:\n");
	printf("  -m <path>       Model weights (.safetensors)\n");
	printf("  -c <path>       Model config (.json)\n");
	printf("  -t <path>       Tokenizer (.json)\n");
	printf("  -f <path>       Fluid state file for persistence\n");
	printf("\n");
	printf("Runtime Options:\n");
	printf("  --mode <mode>   Operation mode: chat, bench, forge (default: chat)\n");
	printf("  --learn         Enable Nested Learning (default: inference only)\n");
	printf("  --steps <n>     Benchmark: number of tokens to generate\n");
	printf("  --threads <n>   Override OMP_NUM_THREADS\n");
	printf("\nForge Mode (Automated Training):\n");
	printf("  Commands via stdin: LEARN <text>, FLUSH <file>, RESET, EXIT\n");
	printf("\n");
	printf("Other:\n");
	printf("  -h, --help      Show this help\n");
	printf("  -v, --version   Show version\n");
	printf("\n");
	printf("Examples:\n");
	printf("  %s -m model.safetensors -c config.json -t tokenizer.json\n", prog);
	printf("  %s --mode bench --steps 100\n", prog);
	printf("  %s --learn -f brain.fluid\n", prog);
	printf("\n");
}

static int	parse_args(int argc, char **argv, t_cli_config *cfg)
{
	int	i;

	/* Defaults */
	cfg->model_path = DEFAULT_MODEL;
	cfg->config_path = DEFAULT_CONFIG;
	cfg->tokenizer_path = DEFAULT_TOKENIZER;
	cfg->fluid_path = NULL;
	cfg->mode = MODE_CHAT;
	cfg->enable_learn = 0;
	cfg->bench_steps = 100;
	cfg->threads = 0;
	cfg->show_help = 0;

	i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
			cfg->model_path = argv[++i];
		else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc)
			cfg->config_path = argv[++i];
		else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
			cfg->tokenizer_path = argv[++i];
		else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc)
			cfg->fluid_path = argv[++i];
		else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
		{
			i++;
			if (strcmp(argv[i], "bench") == 0)
				cfg->mode = MODE_BENCH;
			else if (strcmp(argv[i], "forge") == 0)
				cfg->mode = MODE_FORGE;
			else
				cfg->mode = MODE_CHAT;
		}
		else if (strcmp(argv[i], "--learn") == 0)
			cfg->enable_learn = 1;
		else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
			cfg->bench_steps = atoi(argv[++i]);
		else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
			cfg->threads = atoi(argv[++i]);
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
			cfg->show_help = 1;
		else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0)
		{
			printf("42-engine v%s\n", VERSION);
			return (1);
		}
		else
		{
			fprintf(stderr, "Unknown option: %s\n", argv[i]);
			cfg->show_help = 1;
		}
		i++;
	}
	return (0);
}

/* ============================================================================
** UTF-8 Token Printing (from chat_adaptive.c)
** ============================================================================ */

static int	utf8_char_len(unsigned char c)
{
	if ((c & 0x80) == 0x00) return (1);
	if ((c & 0xC0) == 0x80) return (0);
	if ((c & 0xE0) == 0xC0) return (2);
	if ((c & 0xF0) == 0xE0) return (3);
	if ((c & 0xF8) == 0xF0) return (4);
	return (1);
}

static void	print_token_utf8(t_engine_context *ctx, const char *piece)
{
	int i = 0;
	while (piece[i])
	{
		unsigned char c = (unsigned char)piece[i];
		if (ctx->utf8_len == 0)
		{
			int len = utf8_char_len(c);
			if (len > 1)
			{
				ctx->utf8_buf[ctx->utf8_len++] = c;
				ctx->nl_learn_steps = len;
			}
			else
				putchar(c);
		}
		else
		{
			ctx->utf8_buf[ctx->utf8_len++] = c;
			if (ctx->utf8_len >= ctx->nl_learn_steps)
			{
				for (int j = 0; j < ctx->utf8_len; j++)
					putchar(ctx->utf8_buf[j]);
				ctx->utf8_len = 0;
			}
		}
		i++;
	}
	fflush(stdout);
}

/* ============================================================================
** KV Cache Reset
** ============================================================================ */

static void	reset_kv_caches(t_transformer *t, t_engine_context *ctx)
{
	int	l;

	for (l = 0; l < t->config.n_layers; l++)
	{
		if (t->use_paged_kv)
			paged_kv_reset(&t->paged_kv[l]);
		else
			t->state.kv_cache[l].current_seq_len = 0;
	}
	/* [FIX] Reset LSH index to prevent zombie routing */
	if (t->lsh_index)
		lsh_index_reset((t_lsh_index *)t->lsh_index);
	ctx->session_pos = 0;
	ctx_reset_utf8(ctx);
	ctx_reset_response(ctx);
}

/* ============================================================================
** Chat Mode (Interactive REPL)
** ============================================================================ */

static int	run_chat_mode(t_transformer *t, t_tokenizer *tok, 
				t_engine_context *ctx, const char *fluid_path)
{
	char	input_buf[4096];

	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║           42-BERLIN-ENGINE: Interactive Mode                 ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");
	if (t->nested_learning)
	{
		printf("Commands:\n");
		printf("  LEARN <fact>  - Teach the model a new fact\n");
		printf("  QUERY <text>  - Ask a question\n");
		printf("  RESET         - Clear KV cache\n");
		printf("  SAVE          - Save brain to disk\n");
		printf("  EXIT          - Quit and save\n");
	}
	else
	{
		printf("Commands:\n");
		printf("  <text>        - Generate completion\n");
		printf("  RESET         - Clear KV cache\n");
		printf("  EXIT          - Quit\n");
	}
	printf("\n");

	while (!g_shutdown_requested)
	{
		printf(">> ");
		if (!fgets(input_buf, sizeof(input_buf), stdin))
			break;
		input_buf[strcspn(input_buf, "\n")] = 0;

		if (strcasecmp(input_buf, "EXIT") == 0)
		{
			if (fluid_path)
			{
				if (fluid_save(t, fluid_path) == 0)
					printf("[BRAIN] Saved to %s\n", fluid_path);
			}
			break;
		}

		if (strcasecmp(input_buf, "RESET") == 0)
		{
			reset_kv_caches(t, ctx);
			printf("[RESET] KV cache cleared.\n");
			continue;
		}

		if (strcasecmp(input_buf, "SAVE") == 0 && fluid_path)
		{
			if (fluid_save(t, fluid_path) == 0)
				printf("[BRAIN] Saved to %s\n", fluid_path);
			fluid_print_stats(t);
			continue;
		}

		/* LEARN command (only if learning enabled) */
		if (t->nested_learning && strncasecmp(input_buf, "LEARN ", 6) == 0)
		{
			const char *fact = input_buf + 6;
			printf("[LEARN] Teaching: '%s'\n", fact);

			int *tokens;
			int n_tokens = tokenizer_encode(tok, fact, &tokens);
			if (n_tokens <= 0)
				continue;

			printf("[LEARN] Multi-epoch training (%d epochs)...\n", TRAIN_EPOCHS);

			for (int epoch = 0; epoch < TRAIN_EPOCHS; epoch++)
			{
				reset_kv_caches(t, ctx);
				backward_zero_grads(t);
				for (int i = 0; i < n_tokens; i++)
				{
					transformer_forward(t, tokens[i], ctx->session_pos);
					if (i < n_tokens - 1)
						transformer_backward_step(t, tokens[i + 1], ctx->session_pos);
					ctx->session_pos++;
				}
				backward_apply_grads(t, ADAPTIVE_LR);
			}
			free(tokens);
			reset_kv_caches(t, ctx);
			printf("[LEARN] Done! Fact encoded.\n");
			continue;
		}

		/* QUERY or direct generation */
		const char *query = input_buf;
		if (strncasecmp(input_buf, "QUERY ", 6) == 0)
			query = input_buf + 6;

		reset_kv_caches(t, ctx);
		int old_learn = t->nested_learning;
		t->nested_learning = 0;  /* Disable learning during inference */

		int *tokens;
		int n_tokens = tokenizer_encode(tok, query, &tokens);
		if (n_tokens <= 0)
			continue;

		/* Prefill */
		for (int i = 0; i < n_tokens; i++)
		{
			transformer_forward(t, tokens[i], ctx->session_pos);
			ctx->session_pos++;
		}

		int current_token = tokens[n_tokens - 1];
		free(tokens);

		printf("[ANSWER] ");
		ctx_reset_utf8(ctx);

		int last_token = -1;
		int repeat_count = 0;

		/* Generate */
		for (int gen = 0; gen < MAX_GEN_LEN && !g_shutdown_requested; gen++)
		{
			float *logits = transformer_forward(t, current_token, ctx->session_pos);

			t_tensor logits_t;
			logits_t.data = logits;
			logits_t.size = t->config.vocab_size;
			logits_t.dtype = DTYPE_F32;

			int next_token = sample_argmax(&logits_t);

			if (next_token == tok->eos_id || next_token == 0)
				break;

			if (next_token == last_token)
			{
				repeat_count++;
				if (repeat_count >= 3)
					break;
			}
			else
			{
				repeat_count = 0;
				last_token = next_token;
			}

			const char *piece = tokenizer_decode(tok, next_token);
			if (piece)
				print_token_utf8(ctx, (char *)piece);

			current_token = next_token;
			ctx->session_pos++;

			if (ctx->session_pos >= t->config.seq_len)
				break;
		}
		printf("\n");

		t->nested_learning = old_learn;
	}
	return (0);
}

/* ============================================================================
** Bench Mode (Performance Testing)
** ============================================================================ */

static int	run_bench_mode(t_transformer *t, t_tokenizer *tok,
				t_engine_context *ctx, int steps)
{
	struct timespec	start, end;
	double			elapsed;
	int				*tokens;
	int				n_tokens;

	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║           42-BERLIN-ENGINE: Benchmark Mode                   ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");

	/* Test prompt for prefill */
	const char *test_prompt = 
		"The quick brown fox jumps over the lazy dog. "
		"This is a benchmark prompt to measure throughput.";

	n_tokens = tokenizer_encode(tok, test_prompt, &tokens);
	if (n_tokens <= 0)
	{
		fprintf(stderr, "[BENCH] Failed to encode test prompt\n");
		return (1);
	}

	/* Warmup */
	printf("[BENCH] Warming up...\n");
	reset_kv_caches(t, ctx);
	for (int i = 0; i < n_tokens && i < 10; i++)
	{
		transformer_forward(t, tokens[i], ctx->session_pos);
		ctx->session_pos++;
	}

	/* Prefill benchmark */
	printf("[BENCH] Running prefill benchmark (%d tokens)...\n", n_tokens);
	reset_kv_caches(t, ctx);
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (int i = 0; i < n_tokens; i++)
	{
		transformer_forward(t, tokens[i], ctx->session_pos);
		ctx->session_pos++;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
	printf("[BENCH] Prefill: %d tokens in %.2f ms (%.2f T/s)\n",
		n_tokens, elapsed * 1000, n_tokens / elapsed);

	/* Generation benchmark */
	printf("[BENCH] Running generation benchmark (%d tokens)...\n", steps);
	int current_token = tokens[n_tokens - 1];
	free(tokens);

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (int i = 0; i < steps && !g_shutdown_requested; i++)
	{
		float *logits = transformer_forward(t, current_token, ctx->session_pos);
		t_tensor logits_t = {.data = logits, .size = t->config.vocab_size, .dtype = DTYPE_F32};
		current_token = sample_argmax(&logits_t);
		ctx->session_pos++;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║                        RESULTS                               ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");
	printf("  Generation: %d tokens in %.2f s\n", steps, elapsed);
	printf("  Throughput: %.2f tokens/sec\n", steps / elapsed);
	printf("\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("       GENERATION SPEED: %.2f T/s\n", steps / elapsed);
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("\n");

	return (0);
}

/* ============================================================================
** Forge Mode (Headless Training for Fluid Capsules)
** ============================================================================
** Protocol:
**   LEARN <text>  - Learn the text via teacher forcing
**   FLUSH <file>  - Save current state to .fluid file
**   RESET         - Clear KV cache and LSH index
**   EXIT          - Shutdown
** ============================================================================ */

#define FORGE_LR 0.02f
#define FORGE_EPOCHS 3

static int	run_forge_mode(t_transformer *t, t_tokenizer *tok,
				t_engine_context *ctx)
{
	char	buffer[16384];
	int		*tokens;
	int		n_tokens;
	int		i;
	int		epoch;

	/* Disable stdout buffering for fast pipe communication */
	setvbuf(stdout, NULL, _IONBF, 0);

	/* Signal ready to controller */
	printf("READY\n");

	/* Auto-enable learning in forge mode */
	t->nested_learning = 1;

	while (fgets(buffer, sizeof(buffer), stdin))
	{
		/* Strip newline */
		buffer[strcspn(buffer, "\n")] = 0;

		/* LEARN <text> - Supervised learning on text */
		if (strncmp(buffer, "LEARN ", 6) == 0)
		{
			const char *text = buffer + 6;

			/* Tokenize */
			n_tokens = tokenizer_encode(tok, text, &tokens);
			if (n_tokens <= 1)
			{
				printf("ERROR Too few tokens\n");
				continue;
			}

			/* Multi-epoch teacher forcing */
			for (epoch = 0; epoch < FORGE_EPOCHS; epoch++)
			{
				/* Reset context for clean learning */
				reset_kv_caches(t, ctx);
				backward_zero_grads(t);

				/* Forward + backward for each token pair */
				for (i = 0; i < n_tokens - 1; i++)
				{
					transformer_forward(t, tokens[i], ctx->session_pos);
					transformer_backward_step(t, tokens[i + 1], ctx->session_pos);
					ctx->session_pos++;
				}

				/* Apply accumulated gradients */
				backward_apply_grads(t, FORGE_LR);
			}

			free(tokens);
			reset_kv_caches(t, ctx);
			printf("OK\n");
		}
		/* FLUSH <filename> - Save fluid state */
		else if (strncmp(buffer, "FLUSH ", 6) == 0)
		{
			const char *filename = buffer + 6;
			t_fluid_save_opts opts = {
				.domain = "forged",
				.author = "Fluid-Forge",
				.description = "Auto-trained knowledge capsule",
				.base_model_hash = 0
			};
			if (fluid_save_v2(t, filename, &opts) == 0)
				printf("SAVED %s\n", filename);
			else
				printf("ERROR Save failed\n");
		}
		/* RESET - Clear KV cache and LSH */
		else if (strncmp(buffer, "RESET", 5) == 0)
		{
			reset_kv_caches(t, ctx);
			printf("RESET_OK\n");
		}
		/* EXIT - Shutdown */
		else if (strncmp(buffer, "EXIT", 4) == 0)
		{
			printf("BYE\n");
			break;
		}
		/* Unknown command */
		else
		{
			printf("ERROR Unknown command\n");
		}
	}

	return (0);
}

/* ============================================================================
** Main Entry Point
** ============================================================================ */

int	main(int argc, char **argv)
{
	t_cli_config	cfg;
	t_transformer	t;
	t_tokenizer		tok;
	t_engine_context ctx;
	int				ret;

	/* Parse CLI arguments */
	if (parse_args(argc, argv, &cfg) != 0)
		return (0);
	if (cfg.show_help)
	{
		print_usage(argv[0]);
		return (0);
	}

	/* Set locale and threads */
	setlocale(LC_ALL, "en_US.UTF-8");
	if (cfg.threads > 0)
		omp_set_num_threads(cfg.threads);

	/* Banner */
	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║           42-BERLIN-ENGINE v%s                          ║\n", VERSION);
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");
	printf("[INIT] Model:     %s\n", cfg.model_path);
	printf("[INIT] Config:    %s\n", cfg.config_path);
	printf("[INIT] Tokenizer: %s\n", cfg.tokenizer_path);
	printf("[INIT] Mode:      %s\n", 
		cfg.mode == MODE_BENCH ? "benchmark" : 
		(cfg.mode == MODE_FORGE ? "forge" : "chat"));
	printf("[INIT] Learning:  %s\n", cfg.enable_learn ? "enabled" : "disabled");
	if (cfg.fluid_path)
		printf("[INIT] Fluid:     %s\n", cfg.fluid_path);
	printf("\n");

	/* Initialize transformer */
	printf("[INIT] Loading model...\n");
	if (transformer_init(&t, cfg.model_path, cfg.config_path) != 0)
	{
		fprintf(stderr, "[FATAL] Failed to initialize transformer\n");
		return (1);
	}

	/* Initialize tokenizer */
	if (tokenizer_init(&tok, cfg.tokenizer_path) != 0)
	{
		fprintf(stderr, "[FATAL] Failed to initialize tokenizer\n");
		transformer_free(&t);
		return (1);
	}

	/* Initialize context */
	ctx_init(&ctx);
	t.nested_learning = cfg.enable_learn;
	t.persistent_mode = 1;

	/* Load fluid state if specified */
	if (cfg.fluid_path && access(cfg.fluid_path, F_OK) == 0)
	{
		if (fluid_load(&t, cfg.fluid_path) == 0)
			printf("[BRAIN] Loaded persistent memory from %s\n", cfg.fluid_path);
	}
	else if (cfg.fluid_path)
		printf("[BRAIN] No existing fluid file. Starting fresh.\n");

	/* Setup signal handler for graceful shutdown */
	g_transformer = &t;
	g_fluid_path = cfg.fluid_path;
	signal(SIGINT, signal_handler);

	/* Run selected mode */
	if (cfg.mode == MODE_FORGE)
		printf("[READY] Engine in FORGE mode. Awaiting commands via stdin.\n");
	else
		printf("[READY] Engine initialized. Press Ctrl+C to safely exit.\n");

	if (cfg.mode == MODE_BENCH)
		ret = run_bench_mode(&t, &tok, &ctx, cfg.bench_steps);
	else if (cfg.mode == MODE_FORGE)
		ret = run_forge_mode(&t, &tok, &ctx);
	else
		ret = run_chat_mode(&t, &tok, &ctx, cfg.fluid_path);

	/* [FIX #2] GRACEFUL SHUTDOWN: Save state AFTER main loop exits
	** This is the SAFE place to do IO - not in the signal handler!
	** The signal handler only sets g_shutdown_requested, we react here. */
	if (g_shutdown_requested && g_fluid_path)
	{
		printf("\n[SHUTDOWN] Signal received. Saving state...\n");
		if (fluid_save(&t, g_fluid_path) == 0)
			printf("[SHUTDOWN] Brain saved to %s\n", g_fluid_path);
		printf("[SHUTDOWN] Graceful shutdown complete.\n");
	}

	/* Cleanup */
	transformer_free(&t);
	tokenizer_free(&tok);
	return (ret);
}
