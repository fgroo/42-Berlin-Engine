/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   chat_adaptive.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/17 23:30:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/17 23:30:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
#include "memory/kv_cache.h"
#include "memory/paged.h"
#include "config.h"
#include "engine_context.h"
#include "safe_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <locale.h>
#include <unistd.h>  /* access() */
#include "nested/persistence.h"

#define ADAPTIVE_LR 0.1f  // Aggressive learning for real-time adaptation
#define TRAIN_EPOCHS 5    // Multiple epochs like bench_learn

/*
** UTF-8 Safe Printing (from chat.c)
*/
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
				ctx->nl_learn_steps = len; // Reuse field as expected_len
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

/*
** Reset KV caches (both standard and paged) and position counter
*/
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
	ctx->session_pos = 0;
	ctx_reset_utf8(ctx);
	ctx_reset_response(ctx);
}

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		printf("Usage: %s <model_path> <config_path> <tokenizer_path>\n", argv[0]);
		return (1);
	}

	setlocale(LC_ALL, "en_US.UTF-8");
	printf("Initializing 42-BERLIN-ENGINE [ADAPTIVE MODE]...\n");

	t_transformer t;
	if (transformer_init(&t, argv[1], argv[2]) != 0)
		return (1);

	t_tokenizer tok;
	if (tokenizer_init(&tok, argv[3]) != 0)
		return (1);

	t_engine_context ctx;
	ctx_init(&ctx);

	t.nested_learning = 1;
	t.persistent_mode = 1;

	/* Project Black Box: Load persistent brain on startup */
	if (access("brain.fluid", F_OK) == 0)
	{
		if (fluid_load(&t, "brain.fluid") == 0)
			printf("[BRAIN] Loaded persistent memory.\n");
	}
	else
		printf("[BRAIN] No brain.fluid found - starting fresh.\n");

	char input_buf[4096];
	printf("\n============================================================\n");
	printf("         NESTED LEARNING DEMO - Real-Time Adaptation\n");
	printf("============================================================\n");
	printf("Commands:\n");
	printf("  LEARN <fact>  - Teach the model a new fact\n");
	printf("  QUERY <text>  - Ask a question (model uses learned biases)\n");
	printf("  RESET         - Clear KV cache (biases are retained)\n");
	printf("  SAVE          - Save brain to disk (auto-saves on EXIT)\n");
	printf("  EXIT          - Quit and save brain\n");
	printf("\nExample:\n");
	printf("  LEARN The secret code is 7742\n");
	printf("  QUERY The secret code is\n");
	printf("============================================================\n");

	while (1)
	{
		printf("\n>> ");
		if (!fgets(input_buf, sizeof(input_buf), stdin)) break;
		input_buf[strcspn(input_buf, "\n")] = 0;

		if (strcasecmp(input_buf, "EXIT") == 0)
		{
			/* Project Black Box: Save brain on exit */
			if (fluid_save(&t, "brain.fluid") == 0)
				printf("[BRAIN] Saved persistent memory.\n");
			break;
		}
		
		/* SAVE command: Manual save */
		if (strcasecmp(input_buf, "SAVE") == 0)
		{
			if (fluid_save(&t, "brain.fluid") == 0)
				printf("[BRAIN] Saved to brain.fluid\n");
			fluid_print_stats(&t);
			continue;
		}
		
		if (strcasecmp(input_buf, "RESET") == 0)
		{
			reset_kv_caches(&t, &ctx);
			printf("[RESET] KV cache cleared. Biases retained.\n");
			continue;
		}
		
		/* LEARN command: Multi-epoch training on user-provided fact */
		if (strncasecmp(input_buf, "LEARN ", 6) == 0)
		{
			const char *fact = input_buf + 6;
			printf("[LEARN] Teaching: '%s'\n", fact);
			
			int *tokens;
			int n_tokens = tokenizer_encode(&tok, fact, &tokens);
			if (n_tokens <= 0) continue;

			printf("[LEARN] Multi-epoch training (%d epochs, LR=%.2f)...\n", TRAIN_EPOCHS, ADAPTIVE_LR);
			
			for (int epoch = 0; epoch < TRAIN_EPOCHS; epoch++)
			{
				reset_kv_caches(&t, &ctx);
				
				backward_zero_grads(&t);
				for (int i = 0; i < n_tokens; i++)
				{
					transformer_forward(&t, tokens[i], ctx.session_pos);
					if (i < n_tokens - 1)
						transformer_backward_step(&t, tokens[i + 1], ctx.session_pos);
					ctx.session_pos++;
				}
				backward_apply_grads(&t, ADAPTIVE_LR);
			}
			free(tokens);
			
			reset_kv_caches(&t, &ctx);
			printf("[LEARN] Done! Fact encoded in fluid weights.\n");
			continue;
		}
		
		/* QUERY command: Prefill query then generate with learned biases */
		if (strncasecmp(input_buf, "QUERY ", 6) == 0)
		{
			const char *query = input_buf + 6;
			printf("[QUERY] Asking: '%s'\n", query);
			
			reset_kv_caches(&t, &ctx);
			t.nested_learning = 0; // Disable learning during inference
			
			int *tokens;
			int n_tokens = tokenizer_encode(&tok, query, &tokens);
			if (n_tokens <= 0) continue;

			// Prefill query to build KV cache context
			for (int i = 0; i < n_tokens; i++)
			{
				transformer_forward(&t, tokens[i], ctx.session_pos);
				ctx.session_pos++;
			}
			
			int current_token = tokens[n_tokens - 1];
			free(tokens);

			printf("[ANSWER] ");
			ctx_reset_utf8(&ctx);
			
			int last_token = -1;
			int repeat_count = 0;

			for (int gen = 0; gen < MAX_GEN_LEN; gen++)
			{
				float *logits = transformer_forward(&t, current_token, ctx.session_pos);
				
				t_tensor logits_t;
				logits_t.data = logits;
				logits_t.size = t.config.vocab_size;
				logits_t.dtype = DTYPE_F32;
				
				int next_token = sample_argmax(&logits_t);
				
				if (next_token == tok.eos_id || next_token == 0) break;
				
				/* Repetition detection: stop after 3 consecutive same tokens */
				if (next_token == last_token)
				{
					repeat_count++;
					if (repeat_count >= 3) break;
				}
				else
				{
					repeat_count = 0;
					last_token = next_token;
				}

				const char *piece = tokenizer_decode(&tok, next_token);
				if (piece)
					print_token_utf8(&ctx, (char *)piece);

				current_token = next_token;
				ctx.session_pos++;
				
				if (ctx.session_pos >= t.config.seq_len) break;
			}
			printf("\n");
			
			t.nested_learning = 1; // Re-enable learning
			continue;
		}
		
		printf("[ERROR] Unknown command. Use LEARN, QUERY, RESET, or EXIT.\n");
	}

	transformer_free(&t);
	tokenizer_free(&tok);
	return (0);
}
