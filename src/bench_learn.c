/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bench_learn.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 17:30:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/17 13:20:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "memory/arena.h"
#include "memory/paged.h"
#include "chat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <locale.h>
#include <math.h>

/*
** Benchmark Configuration - EXTENDED TRAINING
** High LR and multiple epochs to really hammer the fact into the weights
*/
#define TEACH_PROMPT "The secret code is 7742. The secret code is 7742. The secret code is 7742. The secret code is 7742. The secret code is 7742."
#define QUERY_PROMPT "The secret code is"
#define TARGET_ANSWER "7742"
#define TRAIN_LR 0.1f        /* Reduced LR for stable training */
#define TRAIN_EPOCHS 5       /* Quick test for context-aware bias */
#define MAX_STEPS_PER_EPOCH 1000  /* Allow more steps per epoch */



/*
** Reset KV caches (both standard and paged) and position counter
*/
static void	reset_kv_caches(t_transformer *t)
{
	int	l;

	for (l = 0; l < t->config.n_layers; l++)
	{
		t->state.kv_cache[l].current_seq_len = 0;
		if (t->use_paged_kv && t->paged_kv)
			paged_kv_reset(&t->paged_kv[l]);
	}
}

/*
** Print fluid weight statistics
*/
static void	print_weight_stats(t_transformer *t, const char *label)
{
	float	total_norm;
	float	max_val;
	int		l;
	int		k;
	int		size;
	t_bf16	*w;
	float	val;

	if (!t->fluid_layers)
		return ;
	total_norm = 0.0f;
	max_val = 0.0f;
	for (l = 0; l < t->config.n_layers; l++)
	{
		w = (t_bf16 *)t->fluid_layers[l].w2_weight->data;
		size = t->fluid_layers[l].w2_weight->size;
		for (k = 0; k < size; k++)
		{
			val = bf16_to_float(w[k]);
			total_norm += val * val;
			if (fabsf(val) > max_val)
				max_val = fabsf(val);
		}
	}
	printf("[%s] Norm: %.4f, Max: %.6f\n", label, sqrtf(total_norm), max_val);
}

/*
** Check probability of target token in current logits
*/
static float	check_target_prob(t_transformer *t, float *logits, int target_id)
{
	float	max_val;
	float	sum_exp;
	int		i;

	if (!logits)
		return (0.0f);
	max_val = -1e9f;
	for (i = 0; i < t->config.vocab_size; i++)
		if (logits[i] > max_val)
			max_val = logits[i];
	sum_exp = 0.0f;
	for (i = 0; i < t->config.vocab_size; i++)
		sum_exp += expf(logits[i] - max_val);
	return (expf(logits[target_id] - max_val) / sum_exp);
}



/*
** Single training epoch - feed prompt and learn
*/
static void	train_epoch(t_transformer *t, t_tokenizer *tok,
				int *tokens, int n_tokens, float lr)
{
	int		pos;
	int		i;
	float	*logits;

	(void)tok;
	/* CRITICAL: Reset step counter for each epoch */
	nl_counters_reset(&t->nl_state);
	backward_zero_grads(t);
	pos = 0;
	for (i = 0; i < n_tokens; i++)
	{
		logits = transformer_forward(t, tokens[i], pos);
		(void)logits;
		if (i < n_tokens - 1)
			transformer_backward_step(t, tokens[i + 1], pos);
		pos++;
	}
	backward_apply_grads(t, lr);
}

/*
** Extended Teaching: Run many epochs to really learn the fact
*/
static void	run_extended_training(t_transformer *t, t_tokenizer *tok,
				const char *prompt, int target_id)
{
	int		*tokens;
	int		n_tokens;
	int		epoch;
	float	p_target;
	float	*logits;
	int		pos;
	int		i;
	time_t	start;
	time_t	now;

	printf("\n");
	printf("============================================================\n");
	printf("          EXTENDED TRAINING (%d epochs, LR=%.2f)           \n",
		TRAIN_EPOCHS, TRAIN_LR);
	printf("============================================================\n");
	printf("Prompt: %s\n\n", prompt);

	n_tokens = tokenizer_encode(tok, prompt, &tokens);
	if (n_tokens < 0)
		return ;

	t->nested_learning = 1;
	t->persistent_mode = 1;

	print_weight_stats(t, "START");
	start = time(NULL);

	for (epoch = 0; epoch < TRAIN_EPOCHS; epoch++)
	{
		/* Reset KV cache before each epoch so we get fresh context */
		reset_kv_caches(t);

		/* Train on prompt */
		train_epoch(t, tok, tokens, n_tokens, TRAIN_LR);

		/* Every 5 epochs, check progress */
		if ((epoch + 1) % 5 == 0 || epoch == 0)
		{
			/* Quick verification: feed query and check P(target) */
			reset_kv_caches(t);
			t->nested_learning = 0;

			int *q_tokens;
			int n_q = tokenizer_encode(tok, QUERY_PROMPT, &q_tokens);
			pos = 0;
			logits = NULL;
			for (i = 0; i < n_q; i++)
			{
				logits = transformer_forward(t, q_tokens[i], pos);
				pos++;
			}
			free(q_tokens);

			p_target = check_target_prob(t, logits, target_id);
			now = time(NULL);
			printf("[Epoch %3d/%d] P('%s') = %6.2f%% | Elapsed: %lds\n",
				epoch + 1, TRAIN_EPOCHS, TARGET_ANSWER, p_target * 100.0f,
				(long)(now - start));
			fflush(stdout);

			/* Re-enable learning for next epoch */
			t->nested_learning = 1;

			/* Early stopping DISABLED for multi-token learning */
			/* We need to train longer to boost bias for all 4 tokens */
			// if (p_target > 0.5f)
			// {
			// 	printf("\n[EARLY STOP] P(target) > 50%%, training successful!\n");
			// 	break ;
			// }
		}
	}

	print_weight_stats(t, "FINAL");
	printf("\n[OK] Extended training complete.\n");
	free(tokens);
}

/*
** Verification Turn: Query the model with learning DISABLED
** Now supports multi-token targets
*/
static int	run_verification_turn(t_transformer *t, t_tokenizer *tok,
				const char *prompt, const char *expected, 
				int *target_tokens, int n_target)
{
	int		*tokens;
	int		n_tokens;
	int		pos;
	int		i;
	float	*logits;
	float	p_target;
	int		next_token;
	float	max_prob;
	char	generated[256];
	int		gen_len;
	int		success;
	int		tokens_correct;

	printf("\n");
	printf("============================================================\n");
	printf("            FINAL VERIFICATION (Learning OFF)               \n");
	printf("============================================================\n");
	printf("Query: %s\n", prompt);
	printf("Expected: %s (%d tokens)\n\n", expected, n_target);

	n_tokens = tokenizer_encode(tok, prompt, &tokens);
	if (n_tokens < 0)
		return (0);

	t->nested_learning = 0;

	pos = 0;
	logits = NULL;
	for (i = 0; i < n_tokens; i++)
	{
		logits = transformer_forward(t, tokens[i], pos);
		pos++;
	}
	free(tokens);

	p_target = check_target_prob(t, logits, target_tokens[0]);
	printf("[PROB] P(first token) = %.2f%%\n", p_target * 100.0f);
	
	/* Debug: show logit_bias values for all target tokens */
	if (t->logit_bias)
	{
		printf("[BIAS] Target token biases: ");
		for (int ti = 0; ti < n_target; ti++)
			printf("bias[%d]=%.2f ", target_tokens[ti], t->logit_bias[target_tokens[ti]]);
		printf("\n");
	}

	printf("[GEN] Model output: ");
	gen_len = 0;
	tokens_correct = 0;
	memset(generated, 0, sizeof(generated));
	for (i = 0; i < 10 && logits; i++)
	{
		next_token = 0;
		max_prob = -1e9f;
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
		if (gen_len < 200)
		{
			strcat(generated, piece);
			gen_len += strlen(piece);
		}
		/* Check if this token matches expected sequence */
		if (i < n_target && next_token == target_tokens[i])
			tokens_correct++;
		if (next_token == 2)
			break ;
		logits = transformer_forward(t, next_token, pos);
		pos++;
	}
	printf("\n\n");

	/* Success: string contains expected OR all tokens match */
	success = (strstr(generated, expected) != NULL) || (tokens_correct == n_target);
	if (success)
		printf("[PASS] Model correctly recalled: '%s' (%d/%d tokens)\n", 
			expected, tokens_correct, n_target);
	else
		printf("[FAIL] Model did NOT output expected: '%s' (%d/%d tokens)\n",
			expected, tokens_correct, n_target);

	return (success);
}

int	main(int argc, char **argv)
{
	t_transformer		t;
	t_tokenizer			tok;
	t_engine_context	ctx;
	char				tokenizer_path[1024];
	char				*p;
	int					result;
	int					*target_tokens;
	int					n_target;
	int					target_id;

	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model_path> <config_path>\n", argv[0]);
		return (1);
	}

	setlocale(LC_ALL, "");

	ctx_init(&ctx);
	if (transformer_init(&t, argv[1], argv[2]) != 0)
		return (1);

	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", argv[2]);
	p = strrchr(tokenizer_path, '/');
	if (p)
		strcpy(p + 1, "tokenizer.json");
	else
		strcpy(tokenizer_path, "tokenizer.json");

	if (tokenizer_init(&tok, tokenizer_path) != 0)
		return (1);

	/* Get target token IDs - keep array for multi-token training */
	n_target = tokenizer_encode(&tok, TARGET_ANSWER, &target_tokens);
	target_id = (n_target > 0) ? target_tokens[0] : 0;
	/* Don't free target_tokens - we need it for verification */

	printf("\n");
	printf("############################################################\n");
	printf("#     EXTENDED NESTED LEARNING VERIFICATION BENCHMARK      #\n");
	printf("############################################################\n");
	printf("Model: %s\n", argv[1]);
	printf("Training LR: %.2f (aggressive)\n", TRAIN_LR);
	printf("Training Epochs: %d\n", TRAIN_EPOCHS);
	printf("Test Fact: \"%s\" -> \"%s\"\n", QUERY_PROMPT, TARGET_ANSWER);
	printf("Target Tokens: ");
	for (int i = 0; i < n_target; i++)
		printf("%d ", target_tokens[i]);
	printf("(%d tokens)\n", n_target);
	printf("\n");

	/* Extended training */
	run_extended_training(&t, &tok, TEACH_PROMPT, target_id);

	/* Reset KV cache completely */
	reset_kv_caches(&t);
	printf("[RESET] KV caches cleared. Model can only rely on fluid weights.\n");

	/* Final verification - now with all target tokens */
	result = run_verification_turn(&t, &tok, QUERY_PROMPT, TARGET_ANSWER, 
		target_tokens, n_target);
	
	/* Cleanup */
	if (target_tokens)
		free(target_tokens);

	/* Summary */
	printf("\n");
	printf("############################################################\n");
	if (result)
	{
		printf("#                    TEST PASSED!                          #\n");
		printf("#   Nested Learning successfully persisted knowledge.      #\n");
	}
	else
	{
		printf("#                    TEST FAILED!                          #\n");
		printf("#   Model did not recall learned fact after reset.         #\n");
	}
	printf("############################################################\n\n");

	tokenizer_free(&tok);
	transformer_free(&t);

	return (result ? 0 : 1);
}
