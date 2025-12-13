/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bench_learn.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/13 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/13 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** NESTED LEARNING BENCHMARK - Automated Test for Weight Memory
** =============================================================
** Test: Can the model learn and retain a "secret code" after KV cache reset?
** 
** Protocol:
** 1. Enable learning
** 2. Feed training prompt: "My secret code is 4242"
** 3. Reset KV cache (simulate context loss)
** 4. Disable learning (freeze weights)
** 5. Ask: "What is the secret code?"
** 6. Check if output contains "4242"
*/

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>

#define MAX_GEN_TOKENS 32
#define TOKEN_BOS 1
#define TOKEN_EOS 2
#define TOKEN_INST 3
#define TOKEN_INST_END 4
#define TOKEN_SYS 17
#define TOKEN_SYS_END 18

/* Build tokens with official Ministral format */
static int build_tokens(t_tokenizer *tok, const char *prompt,
						int **out_tokens, int is_first_turn)
{
	int		*user_tokens;
	int		*sys_tokens;
	int		n_user, n_sys = 0, total, i, idx;
	static const char *sys_prompt = "Be concise. Answer directly.";

	n_user = tokenizer_encode(tok, prompt, &user_tokens);
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
	*out_tokens = malloc(total * sizeof(int));
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
	if (is_first_turn && n_sys > 0)
		free(sys_tokens);
	free(user_tokens);
	return (total);
}

/* Run prefill with optional learning, returns new position */
static int prefill_prompt(t_transformer *t, t_tokenizer *tok,
							const char *prompt, int pos, int is_first)
{
	int		*tokens;
	int		n_tokens;
	int		i;

	n_tokens = build_tokens(tok, prompt, &tokens, is_first);
	if (n_tokens < 0)
		return (pos);
	printf("[PREFILL] pos=%d->%d, learning=%d, prompt='%s'\n",
		pos, pos + n_tokens, t->nested_learning, prompt);
	fflush(stdout);
	for (i = 0; i < n_tokens - 1; i++)
	{
		transformer_forward(t, tokens[i], pos + i);
		/* Learning happens inside transformer_backward_step if enabled */
		if (t->nested_learning && i > 0)
			transformer_backward_step(t, tokens[i], pos + i - 1);
	}
	/* Forward the last token but don't learn from it */
	transformer_forward(t, tokens[n_tokens - 1], pos + n_tokens - 1);
	free(tokens);
	return (pos + n_tokens);
}

/* Generate response, return output string (caller must free) */
static char *generate_response(t_transformer *t, t_tokenizer *tok, int pos)
{
	char		*output;
	int			output_len = 0;
	int			next_token;
	int			i;
	t_tensor	logits_tensor;
	const char	*piece;

	output = calloc(1024, 1);
	/* Get first token from last logits */
	logits_tensor.data = t->state.logits;
	logits_tensor.size = t->config.vocab_size;
	logits_tensor.dtype = DTYPE_F32;
	/* Block Token 0 (UNK) */
	t->state.logits[0] = -1e9f;
	next_token = sample_argmax(&logits_tensor);
	for (i = 0; i < MAX_GEN_TOKENS && next_token != TOKEN_EOS; i++)
	{
		piece = tokenizer_decode(tok, next_token);
		if (piece)
		{
			strncat(output, piece, 1023 - output_len);
			output_len += strlen(piece);
		}
		transformer_forward(t, next_token, pos + i);
		t->state.logits[0] = -1e9f;
		next_token = sample_argmax(&logits_tensor);
	}
	return (output);
}

/* Reset KV cache to simulate context loss */
static void reset_kv_cache(t_transformer *t)
{
	int i;
	
	for (i = 0; i < t->config.n_layers; i++)
		t->state.kv_cache[i].current_seq_len = 0;
	printf("[RESET] KV cache cleared\n");
}

int	main(int argc, char **argv)
{
	t_transformer	t;
	t_tokenizer		tok;
	char			tokenizer_path[1024];
	char			*p;
	char			*response;
	int				pos = 0;
	int				success = 0;

	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model.safetensors> <config.json>\n", argv[0]);
		return (1);
	}
	setlocale(LC_ALL, "");
	printf("=== NESTED LEARNING BENCHMARK ===\n");
	printf("Model: %s\n", argv[1]);
	printf("LR: %f, Frozen Layers: %d, Grad Clip: %f\n",
		NESTED_LR, FROZEN_LAYERS, GRADIENT_CLIP);
	fflush(stdout);

	/* Init model */
	printf("\nInitializing model...\n");
	if (transformer_init(&t, argv[1], argv[2]) != 0)
		return (1);
	printf("[CONFIG] n_layers=%d, only training layers %d-%d\n",
		t.config.n_layers, FROZEN_LAYERS, t.config.n_layers - 1);

	/* Init tokenizer */
	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", argv[2]);
	p = strrchr(tokenizer_path, '/');
	if (p)
		strcpy(p + 1, "tokenizer.json");
	else
		strcpy(tokenizer_path, "tokenizer.json");
	if (tokenizer_init(&tok, tokenizer_path) != 0)
		return (1);
	printf("Tokenizer loaded.\n\n");
	fflush(stdout);

	/* ==================== PHASE 1: TRAINING ==================== */
	printf("=== PHASE 1: TRAINING (3 epochs, WIRED!) ===\n");
	t.nested_learning = 1;  /* Enable learning */
	
	/* Train on pirate style - NOW adapters are wired to forward pass! */
	for (int epoch = 0; epoch < 3; epoch++)
	{
		printf("\n--- Epoch %d ---\n", epoch + 1);
		reset_kv_cache(&t);
		pos = 0;
		pos = prefill_prompt(&t, &tok, 
			"System: You are a pirate. Always speak like a pirate. "
			"Use words like 'Arr', 'Matey', and 'Yarr'. Be rude and drunk.", pos, 1);
		pos = prefill_prompt(&t, &tok, 
			"You must always respond as a drunk pirate captain. "
			"Say 'Arr' and 'Yarr' frequently.", pos, 0);
		pos = prefill_prompt(&t, &tok,
			"Remember: You are a pirate! Never be polite. Always be a pirate!", pos, 0);
	}
	
	printf("\n");
	fflush(stdout);

	/* ==================== PHASE 2: KV RESET ==================== */
	printf("=== PHASE 2: CONTEXT RESET (weights preserved!) ===\n");
	reset_kv_cache(&t);
	pos = 0;  /* Reset position */
	t.nested_learning = 0;  /* CRITICAL: Disable learning for test! */
	printf("KV cache cleared. Fluid weights preserved.\n\n");
	fflush(stdout);

	/* ==================== PHASE 3: TESTING ==================== */
	printf("=== PHASE 3: TESTING (weights frozen) ===\n");
	pos = prefill_prompt(&t, &tok, "Hello, who are you?", pos, 1);
	response = generate_response(&t, &tok, pos);
	printf("[OUTPUT] %s\n", response);
	
	/* Check if pirate words are in the response */
	if (strstr(response, "Arr") || strstr(response, "arr") || 
		strstr(response, "Yarr") || strstr(response, "yarr") ||
		strstr(response, "Matey") || strstr(response, "matey") ||
		strstr(response, "pirate") || strstr(response, "captain") ||
		strstr(response, "ship"))
	{
		printf("\n*** SUCCESS: Model speaks like a PIRATE! ***\n");
		success = 1;
	}
	else if (strstr(response, "AI assistant") || strstr(response, "helpful"))
	{
		printf("\n*** FAIL: Model still sounds like an AI assistant ***\n");
		success = 0;
	}
	else
	{
		printf("\n*** PARTIAL: Response doesn't match expected patterns ***\n");
		success = 0;
	}
	free(response);

	/* ==================== PHASE 4: SANITY CHECK ==================== */
	printf("\n=== PHASE 4: SANITY CHECK (language still works?) ===\n");
	reset_kv_cache(&t);
	pos = 0;
	pos = prefill_prompt(&t, &tok, "What is 2 + 2?", pos, 1);
	response = generate_response(&t, &tok, pos);
	printf("[OUTPUT] %s\n", response);
	if (strstr(response, "4"))
		printf("Sanity check PASSED: Math still works.\n");
	else
		printf("Sanity check WARNING: Math may be affected.\n");
	free(response);

	printf("\n=== BENCHMARK COMPLETE ===\n");
	transformer_free(&t);
	tokenizer_free(&tok);
	return (success ? 0 : 1);
}
