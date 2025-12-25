/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   chat_adaptive_test.c                               :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/17 23:58:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/17 23:58:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
#include "memory/kv_cache.h"
#include "memory/paged.h"
#include "config.h"
#include "engine_context.h"
#include "memory/safe_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>

#define ADAPTIVE_LR 0.1f
#define TRAIN_EPOCHS 10  // More epochs for complete chain learning

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

	/* [PHASE 21] DATA FUSION: No space, direct token link! */
	const char *train_str = "The capital of Germany is42Berlin";
	const char *query_str = "The capital of Germany is";
	
	printf("[TEST] Training on: '%s'\n", train_str);
	printf("[TEST] Multi-epoch learning (%d epochs, LR=%.2f)...\n", TRAIN_EPOCHS, ADAPTIVE_LR);
	
	/* [PHASE 18] Calculate prompt_len: How many tokens are in the query (prompt) */
	/* We only want to learn the transition from prompt end to response start */
	int *prompt_tokens;
	int prompt_len = tokenizer_encode(&tok, query_str, &prompt_tokens);
	t.prompt_len = prompt_len;
	printf("[TEST] Prompt length: %d tokens (learning starts after this)\n", prompt_len);
	
	/* [PHASE 19] DEBUG: Show exact tokenization! */
	printf("[DEBUG] Query tokens: ");
	for (int i = 0; i < prompt_len; i++)
	{
		const char *s = tokenizer_decode(&tok, prompt_tokens[i]);
		printf("[%d]='%s' ", prompt_tokens[i], s ? s : "?");
	}
	printf("\n");
	printf("[DEBUG] LAST QUERY TOKEN: ID %d = '%s'\n", 
		prompt_tokens[prompt_len - 1], 
		tokenizer_decode(&tok, prompt_tokens[prompt_len - 1]));
	free(prompt_tokens);
	
	int *tokens;
	int n_tokens = tokenizer_encode(&tok, train_str, &tokens);
	
	/* [PHASE 19] DEBUG: Show training tokenization! */
	printf("[DEBUG] Train tokens (%d total): ", n_tokens);
	for (int i = 0; i < n_tokens; i++)
	{
		const char *s = tokenizer_decode(&tok, tokens[i]);
		printf("[%d]='%s' ", tokens[i], s ? s : "?");
	}
	printf("\n");
	printf("[DEBUG] Response tokens (pos >= %d): ", prompt_len);
	for (int i = prompt_len; i < n_tokens; i++)
	{
		const char *s = tokenizer_decode(&tok, tokens[i]);
		printf("[%d]='%s' ", tokens[i], s ? s : "?");
	}
	printf("\n");
	
	/* Multi-epoch training like bench_learn */
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
		printf("  [Epoch %d/%d] Complete.\n", epoch + 1, TRAIN_EPOCHS);
	}
	free(tokens);

	printf("[TEST] Resetting KV cache...\n");
	reset_kv_caches(&t, &ctx);

	printf("[TEST] Querying: '%s'\n", query_str);
	n_tokens = tokenizer_encode(&tok, query_str, &tokens);

	// Prefill query - this builds KV cache context matching learned bigrams
	for (int i = 0; i < n_tokens; i++)
	{
		transformer_forward(&t, tokens[i], ctx.session_pos);
		ctx.session_pos++;
	}
	
	int current_token = tokens[n_tokens - 1];
	free(tokens);

	printf("[TEST] AI Response: ");
	char response[256];
	int resp_len = 0;
	int found_42berlin = 0;
	
	for (int gen = 0; gen < 20; gen++)
	{
		float *logits = transformer_forward(&t, current_token, ctx.session_pos);

		t_tensor logits_t;
		logits_t.data = logits;
		logits_t.size = t.config.vocab_size;
		logits_t.dtype = DTYPE_F32;
		
		int next_token = sample_argmax(&logits_t);
		
		if (next_token == tok.eos_id || next_token == 0) break;

		const char *piece = tokenizer_decode(&tok, next_token);
		if (piece)
		{
			printf("%s", piece);
			int len = strlen(piece);
			if (resp_len + len < 255) {
				strcpy(response + resp_len, piece);
				resp_len += len;
			}
		}

		current_token = next_token;
		ctx.session_pos++;
	}
	printf("\n");

	// Check if response contains "42Berlin"
	response[resp_len] = '\0';
	found_42berlin = (strstr(response, "42Berlin") != NULL) || 
	                 (strstr(response, "4") && strstr(response, "2") && strstr(response, "Berlin"));

	if (found_42berlin)
	{
		printf("[SUCCESS] Model correctly recalled '42Berlin'!\n");
	}
	else
	{
		printf("[FAILURE] Model did NOT recall '42Berlin'. Got: '%s'\n", response);
	}

	transformer_free(&t);
	tokenizer_free(&tok);
	return (found_42berlin ? 0 : 1);
}
