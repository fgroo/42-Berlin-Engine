/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   chat.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/14 15:00:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
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

#define MAX_INPUT_LEN 4096
#define MAX_TOKENS 8192

// Stop strings for generation (model often doesn't emit EOS)
static const char *g_stop_strings[] = {
	"</s>",
	"User:",
	"[/INST]",
	"[INST]",
	"\n\nUser",
	NULL
};

// Special token IDs (from tokenizer_config.json)
#define TOKEN_BOS 1
#define TOKEN_EOS 2
#define TOKEN_INST 3
#define TOKEN_INST_END 4
#define TOKEN_SYS 17
#define TOKEN_SYS_END 18
#define TOKEN_THINK 34
#define TOKEN_THINK_END 35

// Trim leading and trailing whitespace in-place
static void	trim_whitespace(char *str)
{
	char	*start;
	char	*end;
	size_t	len;

	if (!str || !str[0])
		return ;
	start = str;
	while (*start && (*start == ' ' || *start == '\t'
			|| *start == '\n' || *start == '\r'))
		start++;
	len = strlen(start);
	if (len == 0)
	{
		str[0] = '\0';
		return ;
	}
	end = start + len - 1;
	while (end > start && (*end == ' ' || *end == '\t'
			|| *end == '\n' || *end == '\r'))
		end--;
	len = end - start + 1;
	memmove(str, start, len);
	str[len] = '\0';
}

// ==================== UTF-8 SAFE PRINT (CONTEXT-BASED) ====================

static int	utf8_char_len(unsigned char c)
{
	if ((c & 0x80) == 0x00)
		return (1);
	if ((c & 0xC0) == 0x80)
		return (0);
	if ((c & 0xE0) == 0xC0)
		return (2);
	if ((c & 0xF0) == 0xE0)
		return (3);
	if ((c & 0xF8) == 0xF0)
		return (4);
	return (1);
}

static void	ctx_print_utf8(t_engine_context *ctx, const char *piece)
{
	int				i;
	int				len;
	int				print_end;
	int				expected;
	int				j;
	unsigned char	c;

	if (!piece || !piece[0])
		return ;
	len = strlen(piece);
	i = 0;
	while (i < len && ctx->utf8_len < CTX_UTF8_BUF_SIZE - 6)
		ctx->utf8_buf[ctx->utf8_len++] = (unsigned char)piece[i++];
	print_end = 0;
	i = 0;
	while (i < ctx->utf8_len)
	{
		c = ctx->utf8_buf[i];
		expected = utf8_char_len(c);
		if (expected == 0)
		{
			i++;
			print_end = i;
			continue ;
		}
		if (i + expected <= ctx->utf8_len)
		{
			int valid = 1;
			for (j = 1; j < expected && valid; j++)
				if ((ctx->utf8_buf[i + j] & 0xC0) != 0x80)
					valid = 0;
			if (valid)
			{
				i += expected;
				print_end = i;
			}
			else
			{
				i++;
				print_end = i;
			}
		}
		else
			break ;
	}
	if (print_end > 0)
	{
		ctx->utf8_buf[print_end] = '\0';
		printf("%s", (char *)ctx->utf8_buf);
		fflush(stdout);
		if (print_end < ctx->utf8_len)
		{
			for (i = 0; i < ctx->utf8_len - print_end; i++)
				ctx->utf8_buf[i] = ctx->utf8_buf[print_end + i];
			ctx->utf8_len -= print_end;
		}
		else
			ctx->utf8_len = 0;
	}
}

static void	ctx_flush_utf8(t_engine_context *ctx)
{
	if (ctx->utf8_len > 0)
	{
		ctx->utf8_buf[ctx->utf8_len] = '\0';
		printf("%s", (char *)ctx->utf8_buf);
		fflush(stdout);
		ctx->utf8_len = 0;
	}
}

// ==================== TOKEN BUILDING ====================

static int	build_chat_tokens(t_tokenizer *tok, const char *user_input,
				int **out_tokens, int is_first_turn, int raw_mode)
{
	int			*user_tokens;
	int			*sys_tokens;
	int			n_user;
	int			n_sys;
	int			total;
	int			i;
	int			idx;
	char		*spaced_input;
	static const char *sys_prompt = "Be concise. Answer directly with the result.";

	user_tokens = NULL;
	sys_tokens = NULL;
	n_sys = 0;
	
	if (raw_mode)
	{
		n_user = tokenizer_encode(tok, user_input, &user_tokens);
		if (n_user < 0) return (-1);
		*out_tokens = user_tokens;
		return (n_user);
	}

	spaced_input = xmalloc(strlen(user_input) + 2);
	spaced_input[0] = ' ';
	strcpy(spaced_input + 1, user_input);
	n_user = tokenizer_encode(tok, spaced_input, &user_tokens);
	free(spaced_input);
	if (n_user < 0)
		return (-1);
	if (is_first_turn)
	{
		n_sys = tokenizer_encode(tok, sys_prompt, &sys_tokens);
		if (n_sys < 0)
			n_sys = 0;
	}
	if (is_first_turn)
		total = 1 + 1 + n_sys + 1 + 1 + n_user + 1;
	else
		total = 1 + n_user + 1;
	*out_tokens = xmalloc(total * sizeof(int));
	idx = 0;
	if (is_first_turn)
	{
		(*out_tokens)[idx++] = TOKEN_BOS;
		(*out_tokens)[idx++] = TOKEN_SYS;
		for (i = 0; i < n_sys; i++)
			(*out_tokens)[idx++] = sys_tokens[i];
		(*out_tokens)[idx++] = TOKEN_SYS_END;
	}
	(*out_tokens)[idx++] = TOKEN_INST;
	for (i = 0; i < n_user; i++)
		(*out_tokens)[idx++] = user_tokens[i];
	(*out_tokens)[idx++] = TOKEN_INST_END;
	if (sys_tokens)
		free(sys_tokens);
	free(user_tokens);
	return (total);
}

// ==================== STOP STRING CHECK ====================

static int	check_stop_string(const char *output)
{
	int	i;

	i = 0;
	while (g_stop_strings[i])
	{
		if (strstr(output, g_stop_strings[i]))
			return (1);
		i++;
	}
	return (0);
}

// ==================== RESPONSE BUFFER (SAFE) ====================

static void	ctx_append_response(t_engine_context *ctx, const char *piece)
{
	size_t	len;

	len = (size_t)ctx->response_len;
	safe_strcat(ctx->response_buf, CTX_RESPONSE_BUF_SIZE, piece, &len);
	ctx->response_len = (int)len;
}

// ==================== GENERATION ====================

static int	run_generation(t_transformer *t, t_tokenizer *tok,
				const char *input_text, const char *expected_answer,
				t_arena *sampler_arena, t_engine_context *ctx)
{
	int		*tokens;
	int		is_first_turn;
	int		n_tokens;
	int		next_token;
	int		pos;
	int		is_thinking;
	int		stop_hit;
	int		generated_tokens[MAX_GEN_LEN];
	int		n_generated;
	int		i;

	tokens = NULL;
	is_first_turn = (ctx->session_pos == 0) ? 1 : 0;
	n_tokens = build_chat_tokens(tok, input_text, &tokens, is_first_turn, t->raw_mode);
	if (n_tokens < 0)
		return (0);
	printf("[DEBUG] Turn %s, pos=%d->%d, tokens=%d\n",
		is_first_turn ? "FIRST" : "CONT",
		ctx->session_pos, ctx->session_pos + n_tokens, n_tokens);
	printf("[DEBUG] Tokens: [%d, %d, %d, %d, %d ... LAST=%d]\n",
		tokens[0], tokens[1], n_tokens > 2 ? tokens[2] : -1,
		n_tokens > 3 ? tokens[3] : -1, n_tokens > 4 ? tokens[4] : -1,
		tokens[n_tokens - 1]);
	fflush(stdout);
	if (expected_answer)
		printf("Sanity Check: %s\n", input_text);
	else
		printf("\033[90m[Thinking] ");
	fflush(stdout);

	// CRITICAL: Reset ALL per-turn learning counters!
	// Without this, counters keep incrementing across turns
	nl_counters_reset(&t->nl_state);  // CAS-based atomic reset (Phase 2)

	// NESTED LEARNING: Zero FP32 accumulators at start of turn
	backward_zero_grads(t);

	// Prefill: Use batched prefill for efficiency
	// For learning, we still need to process tokens sequentially for backprop
	if (t->nested_learning)
	{
		// Sequential path with backward steps (learning enabled)
		for (i = 0; i < n_tokens - 1; i++)
		{
			transformer_forward(t, tokens[i], ctx->session_pos + i);
			if (i == 0)
			{
				float max_val = -INFINITY;
				for (int v = 0; v < t->config.vocab_size; v++)
					if (t->state.logits[v] > max_val)
						max_val = t->state.logits[v];
				if (isnan(max_val))
				{
					printf("FATAL: Logits contain NaN! Aborting.\n");
					free(tokens);
					return (0);
				}
			}
			// NESTED LEARNING with global gradient norm clipping
			transformer_backward_step(t, tokens[i + 1], ctx->session_pos + i);
		}
		// Apply accumulated gradients at end of prefill
		backward_apply_grads(t, t->nested_lr);
	}
	else
	{
		// Sequential prefill (same as learning, but without backward)
		// forward_prefill_batch has bugs, using proven path instead
		for (i = 0; i < n_tokens - 1; i++)
			transformer_forward(t, tokens[i], ctx->session_pos + i);
	}

	next_token = tokens[n_tokens - 1];
	pos = ctx->session_pos + n_tokens - 1;
	is_thinking = 1;
	stop_hit = 0;
	n_generated = 0;
	ctx_reset_response(ctx);

	for (i = 0; i < MAX_GEN_LEN; i++)
	{
		transformer_forward(t, next_token, pos);
		t_tensor logits_tensor;
		logits_tensor.data = t->state.logits;
		logits_tensor.size = t->config.vocab_size;
		logits_tensor.dtype = DTYPE_F32;

		// Repetition penalty
		{
			float *logits = t->state.logits;
			for (int j = 0; j < n_tokens; j++)
			{
				if (logits[tokens[j]] > 0)
					logits[tokens[j]] /= REPETITION_PENALTY;
				else
					logits[tokens[j]] *= REPETITION_PENALTY;
			}
			for (int j = 0; j < n_generated; j++)
			{
				if (logits[generated_tokens[j]] > 0)
					logits[generated_tokens[j]] /= REPETITION_PENALTY;
				else
					logits[generated_tokens[j]] *= REPETITION_PENALTY;
			}
		}

		// Block UNK token
		t->state.logits[0] = -1e9f;

		next_token = sample_argmax(&logits_tensor);
		arena_reset(sampler_arena);

		if (n_generated < MAX_GEN_LEN)
			generated_tokens[n_generated++] = next_token;

		if (next_token == TOKEN_EOS || stop_hit)
			break ;

		if (next_token == TOKEN_THINK_END && is_thinking)
		{
			is_thinking = 0;
			ctx_flush_utf8(ctx);
			if (!expected_answer)
				printf("\033[0m\n[Answer] ");
			fflush(stdout);
			pos++;
			continue ;
		}

		const char *piece = tokenizer_decode(tok, next_token);
		if (piece)
		{
			if (!is_thinking && expected_answer)
				ctx_append_response(ctx, piece);
			if (!expected_answer)
			{
				ctx_print_utf8(ctx, piece);
				ctx_append_response(ctx, piece);
				if (check_stop_string(ctx->response_buf))
					stop_hit = 1;
			}
		}
		pos++;
	}

	ctx_flush_utf8(ctx);
	if (!expected_answer)
		printf("\033[0m\n");

	ctx->session_pos = pos + 1;
	free(tokens);

	if (!expected_answer && t->nested_learning)
	{
		printf("[State] Fluid Weights UPDATED. Context preserved.\n");
		if (!t->persistent_mode)
		{
			for (int l = 0; l < t->config.n_layers && t->fluid_layers; l++)
			{
				if (t->fluid_layers[l].w2_weight && t->fluid_layers[l].w2_weight->data)
					memset(t->fluid_layers[l].w2_weight->data, 0,
						t->fluid_layers[l].w2_weight->size * sizeof(uint16_t));
			}
			printf("[State] Fluid Weights RESET (transient mode).\n");
		}
		else
		{
			printf("[State] Fluid Weights PERSISTED (persistent mode).\n");
		}
	}

	if (expected_answer)
	{
		printf("Output: %s\n", ctx->response_buf);
		if (strstr(ctx->response_buf, expected_answer))
			return (1);
		return (0);
	}
	return (1);
}

// ==================== MAIN ====================

int	main(int argc, char **argv)
{
	t_transformer		t;
	t_tokenizer			tok;
	t_arena				sampler_arena;
	t_engine_context	ctx;
	char				input[MAX_INPUT_LEN];
	char				tokenizer_path[1024];
	char				*p;

	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model_path> <config_path>\n", argv[0]);
		return (1);
	}
	if (!setlocale(LC_ALL, "en_US.UTF-8"))
		if (!setlocale(LC_ALL, "C.UTF-8"))
			setlocale(LC_ALL, "");

	// Initialize context (replaces all globals!)
	ctx_init(&ctx);

	printf("Initializing model...\n");
	if (transformer_init(&t, argv[1], argv[2]) != 0)
	{
		fprintf(stderr, "Failed to initialize transformer\n");
		return (1);
	}

	printf("Initializing tokenizer...\n");
	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", argv[2]);
	p = strrchr(tokenizer_path, '/');
	if (p)
		strcpy(p + 1, "tokenizer.json");
	else
		strcpy(tokenizer_path, "tokenizer.json");
	if (tokenizer_init(&tok, tokenizer_path) != 0)
	{
		fprintf(stderr, "Failed to initialize tokenizer from %s\n",
			tokenizer_path);
		return (1);
	}

	arena_init(&sampler_arena, 4 * 1024 * 1024);

	printf("Chat initialized. Type 'exit' to quit.\n");
	printf("Commands: 'learn', 'nolearn', 'persist', 'transient', 'reset', 'raw', 'chat'\n");

	while (1)
	{
		printf("\nUser: ");
		if (!fgets(input, sizeof(input), stdin))
			break ;
		input[strcspn(input, "\n")] = 0;
		trim_whitespace(input);
		if (input[0] == '\0')
			continue ;
		if (strcmp(input, "exit") == 0)
			break ;
		if (strcmp(input, "learn") == 0)
		{
			t.nested_learning = 1;
			printf("[MODE] Learning ENABLED.\n");
			continue ;
		}
		if (strcmp(input, "persist") == 0)
		{
			t.persistent_mode = 1;
			printf("[MODE] Persistent Learning ENABLED. Weights will NOT reset.\n");
			continue ;
		}
		if (strcmp(input, "transient") == 0)
		{
			t.persistent_mode = 0;
			// Reset immediately
			for (int l = 0; l < t.config.n_layers && t.fluid_layers; l++)
			{
				if (t.fluid_layers[l].w2_weight && t.fluid_layers[l].w2_weight->data)
					memset(t.fluid_layers[l].w2_weight->data, 0,
						t.fluid_layers[l].w2_weight->size * sizeof(uint16_t));
			}
			printf("[MODE] Transient Learning ENABLED. Weights reset after turn.\n");
			continue ;
		}

		if (strcmp(input, "nolearn") == 0)
		{
			t.nested_learning = 0;
			printf("[MODE] Learning DISABLED.\n");
			continue ;
		}
		if (strcmp(input, "reset") == 0)
		{
			ctx_reset_conversation(&ctx);
			for (int l = 0; l < t.config.n_layers; l++)
				t.state.kv_cache[l].current_seq_len = 0;
			// Reset paged KV cache if using paged mode
			if (t.use_paged_kv && t.paged_kv)
			{
				for (int l = 0; l < t.config.n_layers; l++)
					paged_kv_reset(&t.paged_kv[l]);
			}
			// Note: Fluid weights are NOT reset here to preserve persist mode learning
			printf("[MODE] Conversation and KV Cache RESET. Fluid weights preserved.\n");
			continue ;
		}
		if (strcmp(input, "raw") == 0)
		{
			t.raw_mode = 1;
			printf("[MODE] Raw Completion ENABLED (No Chat Template).\n");
			continue ;
		}
		if (strcmp(input, "chat") == 0)
		{
			t.raw_mode = 0;
			printf("[MODE] Chat Template ENABLED.\n");
			continue ;
		}
		run_generation(&t, &tok, input, NULL, &sampler_arena, &ctx);
	}

	tokenizer_free(&tok);
	arena_free(&sampler_arena);
	transformer_free(&t);
	return (0);
}
