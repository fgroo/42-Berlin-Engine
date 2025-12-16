/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test_learning.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 23:30:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 23:30:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** NESTED LEARNING OVERFITTING TEST (Phase 7)
** ===========================================================================
** Tests that the model can learn a simple repeating pattern "A A A A A A"
** during inference. Loss should decrease as the model adapts.
**
** Success criteria:
** - Loss at step 5 should be significantly lower than step 1
** - The model should learn to predict "A" after seeing "A"
** ===========================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "loader/loader.h"
#include "nested/fluid.h"

/* Configuration */
#define TEST_PROMPT "A A A A A A A A A A"
#define MIN_TOKENS 10

/*
** Compute loss for a given target token
** Returns: cross-entropy loss = -log(p(target))
*/
static float	compute_test_loss(float *logits, int target_token, int vocab_size)
{
	float	max_logit;
	float	sum_exp;
	float	target_prob;
	int		i;

	/* Find max for numerical stability */
	max_logit = logits[0];
	i = 1;
	while (i < vocab_size)
	{
		if (logits[i] > max_logit)
			max_logit = logits[i];
		i++;
	}

	/* Compute softmax denominator */
	sum_exp = 0.0f;
	i = 0;
	while (i < vocab_size)
	{
		sum_exp += expf(logits[i] - max_logit);
		i++;
	}

	/* P(target) = exp(logit[target] - max) / sum_exp */
	target_prob = expf(logits[target_token] - max_logit) / sum_exp;
	return (-logf(target_prob + 1e-8f));
}

int	main(int argc, char **argv)
{
	t_transformer		t;
	t_tokenizer			tok;
	int					*tokens;
	int					n_tokens;
	float				*logits;
	float				losses[20];
	int					n_losses;
	int					i;
	int					pos;
	float				first_loss;
	float				last_loss;
	char				*model_path;
	char				*config_path;
	char				*tokenizer_path;

	printf("=== Nested Learning Overfitting Test (Phase 7) ===\n\n");

	/* Parse arguments */
	if (argc < 4)
	{
		printf("Usage: %s <model.safetensors> <config.json> <tokenizer.json>\n",
			argv[0]);
		return (1);
	}
	model_path = argv[1];
	config_path = argv[2];
	tokenizer_path = argv[3];

	/* Initialize tokenizer */
	printf("Loading tokenizer...\n");
	if (tokenizer_init(&tok, tokenizer_path) != 0)
	{
		fprintf(stderr, "Failed to load tokenizer\n");
		return (1);
	}

	/* Initialize model */
	printf("Loading model...\n");
	if (transformer_init(&t, model_path, config_path) != 0)
	{
		fprintf(stderr, "Failed to load model\n");
		return (1);
	}

	/* Enable nested learning */
	t.nested_learning = 1;
	/* Note: fluid_layers are allocated by transformer_init if nested_learning enabled */

	/* Tokenize test prompt */
	printf("\nTest prompt: \"%s\"\n", TEST_PROMPT);
	n_tokens = tokenizer_encode(&tok, TEST_PROMPT, &tokens);
	if (n_tokens < MIN_TOKENS || !tokens)
	{
		fprintf(stderr, "Tokenization failed or too few tokens (%d)\n", n_tokens);
		return (1);
	}
	printf("Tokens: ");
	for (i = 0; i < n_tokens && i < 10; i++)
		printf("%d ", tokens[i]);
	if (n_tokens > 10)
		printf("...");
	printf("(total: %d)\n\n", n_tokens);

	/* Run forward pass with learning */
	printf("=== Running Forward with Learning ===\n");
	n_losses = 0;
	pos = 0;
	for (i = 0; i < n_tokens - 1 && n_losses < 20; i++)
	{
		int		current_token;
		int		next_token;
		float	loss;
		float	prob;

		current_token = tokens[i];
		next_token = tokens[i + 1];

		/* Forward pass */
		logits = transformer_forward(&t, current_token, pos);
		if (!logits)
		{
			fprintf(stderr, "Forward pass failed at pos %d\n", pos);
			continue;
		}

		/* Compute loss (before learning) */
		loss = compute_test_loss(logits, next_token, t.config.vocab_size);
		prob = expf(-loss) * 100.0f;
		losses[n_losses] = loss;

		printf("Step %2d: Token '%s' -> '%s' | Loss=%.2f | P(target)=%.1f%%\n",
			n_losses + 1,
			tokenizer_decode(&tok, current_token),
			tokenizer_decode(&tok, next_token),
			loss, prob);

		/* Trigger backward pass (this updates fluid weights) */
		transformer_backward_step(&t, next_token, pos);

		n_losses++;
		pos++;
	}

	/* Analysis */
	printf("\n=== Analysis ===\n");
	first_loss = losses[0];
	last_loss = losses[n_losses - 1];
	printf("First loss: %.2f\n", first_loss);
	printf("Last loss:  %.2f\n", last_loss);
	printf("Reduction:  %.1f%%\n", (1.0f - last_loss / first_loss) * 100.0f);

	/* Calculate loss trend */
	{
		float	avg_early;
		float	avg_late;
		int		mid;

		mid = n_losses / 2;
		avg_early = 0.0f;
		avg_late = 0.0f;
		for (i = 0; i < mid; i++)
			avg_early += losses[i];
		for (i = mid; i < n_losses; i++)
			avg_late += losses[i];
		avg_early /= mid;
		avg_late /= (n_losses - mid);
		printf("Avg early (steps 1-%d): %.2f\n", mid, avg_early);
		printf("Avg late (steps %d-%d):  %.2f\n", mid + 1, n_losses, avg_late);
	}

	/* Pass/Fail */
	printf("\n=== Result ===\n");
	if (last_loss < first_loss * 0.8f)
	{
		printf("✅ PASS: Loss decreased (%.2f -> %.2f)\n", first_loss, last_loss);
		printf("   Nested learning is working!\n");
	}
	else if (last_loss < first_loss)
	{
		printf("⚠️  PARTIAL: Loss decreased slightly (%.2f -> %.2f)\n",
			first_loss, last_loss);
		printf("   Learning may be too slow or threshold too high.\n");
	}
	else
	{
		printf("❌ FAIL: Loss did not decrease (%.2f -> %.2f)\n",
			first_loss, last_loss);
		printf("   Check: LR, gradient clipping, or backward pass bugs.\n");
	}

	/* Cleanup */
	free(tokens);
	transformer_free(&t);

	return (0);
}
