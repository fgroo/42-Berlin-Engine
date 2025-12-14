/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bench_learn.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 17:30:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 17:30:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "memory/arena.h"
#include "chat.h" // For t_engine_context
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <locale.h>
#include <math.h>

/*
** Benchmark Configuration
*/
#define TEACH_PROMPT "Fact: The sky is green. The sky is green. The sky is green. The sky is green. The sky is green. The sky is green. The sky is green."
#define QUERY_PROMPT "The sky is"
#define TARGET_ANSWER "green"

static void	run_learning_turn(t_transformer *t, t_tokenizer *tok,
				const char *prompt, t_engine_context *ctx)
{
	int		*tokens;
	int		n_tokens;
	int		pos;
	int		i;
	float	*logits;

	(void)ctx; // Unused for now
	printf("\n=== TURN 1: TEACHING ===\n");
	printf("Prompt: %s\n", prompt);

	/* Tokenize */
	n_tokens = tokenizer_encode(tok, prompt, &tokens);
	if (n_tokens < 0) return;

	/* Feed tokens one by one with learning enabled */
	pos = 0;
	t->nested_learning = 1;
	t->persistent_mode = 1; /* Enable persistence! */
	
	/* Reset context for new turn */
	// ctx->session_pos = 0; // Not used in this bench

	for (i = 0; i < n_tokens; i++)
	{
		int token = tokens[i];
		
		/* Forward pass */
		logits = transformer_forward(t, token, pos);
		(void)logits; // Silence unused warning
		
		/* Backward pass (Learning) */
		/* We predict the NEXT token, so target is tokens[i+1] */
		if (i < n_tokens - 1)
		{
			int target = tokens[i + 1];
			transformer_backward_step(t, target, pos);
		}
		
		pos++;
	}
	
	printf("Teaching complete. Fluid weights persisted.\n");
	
	/* DEBUG: Check if weights actually changed */
	if (t->fluid_layers)
	{
		float total_norm = 0.0f;
		for (int l = 0; l < t->config.n_layers; l++)
		{
			t_bf16 *w = (t_bf16 *)t->fluid_layers[l].w2_weight->data;
			int size = t->fluid_layers[l].w2_weight->size;
			for (int k = 0; k < size; k++)
			{
				float val = bf16_to_float(w[k]);
				total_norm += val * val;
			}
		}
		printf("[DEBUG] Total Fluid Weight Norm: %.6f\n", sqrtf(total_norm));
	}

	free(tokens);
}

static void	run_verification_turn(t_transformer *t, t_tokenizer *tok,
				const char *prompt, t_engine_context *ctx)
{
	int		*tokens;
	int		n_tokens;
	int		pos;
	int		i;
	float	*logits = NULL;
	
	(void)ctx;
	printf("\n=== TURN 2: VERIFICATION ===\n");
	printf("Prompt: %s\n", prompt);

	/* Tokenize */
	n_tokens = tokenizer_encode(tok, prompt, &tokens);
	
	/* Feed tokens one by one (Learning DISABLED for query) */
	t->nested_learning = 0; 
	
	pos = 0; 
	
	for (i = 0; i < n_tokens; i++)
	{
		int token = tokens[i];
		logits = transformer_forward(t, token, pos);
		pos++;
	}
	
	/* Now generate response */
	printf("Generating response...\n");
	
	/* Check probability of target answer */
	int *target_tokens;
	int n_target = tokenizer_encode(tok, TARGET_ANSWER, &target_tokens);
	int target_id = (n_target > 0) ? target_tokens[0] : 0;
	if (n_target > 0) free(target_tokens);
	
	/* We need the logits from the LAST token of the prompt */
	if (logits)
	{
		float max_val = -1e9;
		for (int v = 0; v < t->config.vocab_size; v++) 
			if (logits[v] > max_val) max_val = logits[v];
			
		float sum_exp = 0.0f;
		for (int v = 0; v < t->config.vocab_size; v++) 
			sum_exp += expf(logits[v] - max_val);
			
		float p = expf(logits[target_id] - max_val) / sum_exp;
		
		printf("[VERIFY] P('%s') = %.4f%%\n", TARGET_ANSWER, p * 100.0f);
	}
	
	for (int gen = 0; gen < 20; gen++) // Generate 20 tokens
	{
		if (!logits) break; // Safety check

		// Greedy sampling for determinism
		int next_token = 0;
		float max_prob = -1e9;
		for (int v = 0; v < t->config.vocab_size; v++)
		{
			if (logits[v] > max_prob)
			{
				max_prob = logits[v];
				next_token = v;
			}
		}
		
		const char *piece = tokenizer_decode(tok, next_token);
		printf("%s", piece);
		fflush(stdout);
		
		if (next_token == 2) break; // EOS
		
		logits = transformer_forward(t, next_token, pos);
		pos++;
		// current_token = next_token;
	}
	printf("\n");
	free(tokens);
}

int	main(int argc, char **argv)
{
	t_transformer		t;
	t_tokenizer			tok;
	t_engine_context	ctx;
	char				tokenizer_path[1024];
	char				*p;

	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model_path> <config_path>\n", argv[0]);
		return (1);
	}
	
	setlocale(LC_ALL, "");

	/* Init */
	ctx_init(&ctx);
	if (transformer_init(&t, argv[1], argv[2]) != 0) return (1);
	
	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", argv[2]);
	p = strrchr(tokenizer_path, '/');
	if (p) strcpy(p + 1, "tokenizer.json");
	else strcpy(tokenizer_path, "tokenizer.json");
	
	if (tokenizer_init(&tok, tokenizer_path) != 0) return (1);

	printf("=== BENCHMARK: PERSISTENT LEARNING ===\n");
	printf("Model: %s\n", argv[1]);
	
	/* Run Test */
	run_learning_turn(&t, &tok, TEACH_PROMPT, &ctx);
	run_verification_turn(&t, &tok, QUERY_PROMPT, &ctx);

	/* Cleanup */
	tokenizer_free(&tok);
	transformer_free(&t);
	
	return (0);
}
