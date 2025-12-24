/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test_pulse.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity                                +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/23 17:00:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/23 17:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** PULSE TEST: Does the Model Learn?
** ===========================================================================
** Single-step overfit test to verify gradient flow through the Fluid layers.
** 
** Procedure:
** 1. Forward pass with input prompt → get logits → compute initial loss
** 2. Backward pass forcing target token → accumulate gradients
** 3. Apply gradients (SGD step with high LR)
** 4. Forward pass again → compute new loss
** 5. Verify: new_loss < initial_loss (the model adapted!)
** ===========================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/inference/inference.h"
#include "../src/tokenizer/tokenizer.h"

/* Cross-entropy loss: -log(P(target)) */
static float compute_ce_loss(float *logits, int vocab_size, int target)
{
	/* Compute softmax for target only (stable version) */
	float max_logit = logits[0];
	for (int i = 1; i < vocab_size; i++)
		if (logits[i] > max_logit)
			max_logit = logits[i];
	
	float sum_exp = 0.0f;
	for (int i = 0; i < vocab_size; i++)
		sum_exp += expf(logits[i] - max_logit);
	
	float log_softmax = (logits[target] - max_logit) - logf(sum_exp);
	return -log_softmax;  /* Cross-entropy is negative log likelihood */
}

int main(int argc, char **argv)
{
	t_transformer t;
	t_tokenizer tok;
	int *tokens = NULL;
	int n_tokens;
	float loss_initial, loss_post;
	int pos;

	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║             42-BERLIN-ENGINE: PULSE TEST                     ║\n");
	printf("║               Does the model learn?                          ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");

	if (argc < 3)
	{
		fprintf(stderr, "Usage: %s <model.safetensors> <tokenizer.json> [config.json]\n", argv[0]);
		return 1;
	}

	/* Derive config path from model path if not provided */
	const char *config_path = (argc >= 4) ? argv[3] : NULL;
	char derived_config[512];
	if (!config_path)
	{
		/* Try to find config.json in same directory as model */
		const char *model_path = argv[1];
		const char *last_slash = strrchr(model_path, '/');
		if (last_slash)
		{
			int dir_len = (int)(last_slash - model_path);
			snprintf(derived_config, sizeof(derived_config), "%.*s/config.json", dir_len, model_path);
		}
		else
		{
			/* Model in current directory, try dummy_config.json */
			snprintf(derived_config, sizeof(derived_config), "dummy_config.json");
		}
		config_path = derived_config;
	}

	/* 1. Initialize model and tokenizer */
	printf("[INIT] Loading model: %s\n", argv[1]);
	printf("[INIT] Loading config: %s\n", config_path);
	if (transformer_init(&t, argv[1], config_path) != 0)
	{
		fprintf(stderr, "FATAL: Failed to load model\n");
		return 1;
	}
	
	printf("[INIT] Loading tokenizer: %s\n", argv[2]);
	if (tokenizer_init(&tok, argv[2]) != 0)
	{
		fprintf(stderr, "FATAL: Failed to load tokenizer\n");
		transformer_free(&t);
		return 1;
	}

	/* Enable nested learning with aggressive LR for testing */
	t.nested_learning = 1;
	t.nested_lr = 0.1f;  /* Very high LR for single-step overfit */
	printf("[CONFIG] Nested Learning: ON (LR=%.3f)\n\n", t.nested_lr);

	/* 2. Encode test prompt */
	const char *prompt = "The answer to life, the universe, and everything is";
	printf("[PROMPT] \"%s\"\n", prompt);
	
	n_tokens = tokenizer_encode(&tok, prompt, &tokens);
	if (!tokens || n_tokens == 0)
	{
		fprintf(stderr, "FATAL: Tokenization failed\n");
		transformer_free(&t);
		return 1;
	}
	printf("[TOKENS] Encoded %d tokens\n", n_tokens);

	/* 3. Forward pass: process all tokens, get logits for next token */
	printf("\n[FORWARD PASS 1] Initial (Frozen + Fluid zeros)...\n");
	backward_zero_grads(&t);  /* Ensure clean gradient state */
	
	float *logits = NULL;
	for (pos = 0; pos < n_tokens; pos++)
	{
		logits = transformer_forward(&t, tokens[pos], pos);
	}
	
	/* Target: We want the model to predict "42" */
	int target_token = 3705;  /* Common ID for "42" in many tokenizers */
	/* Try to find actual token ID for "42" */
	int *target_enc = NULL;
	int target_len = tokenizer_encode(&tok, "42", &target_enc);
	if (target_enc && target_len > 0)
	{
		target_token = target_enc[0];
		free(target_enc);
	}
	printf("[TARGET] Token ID for '42': %d\n", target_token);
	
	loss_initial = compute_ce_loss(logits, t.config.vocab_size, target_token);
	printf("[LOSS] Initial: %.4f (P(42)=%.2e)\n", 
		loss_initial, expf(-loss_initial));

	/* 4. Backward pass: force the target token */
	printf("\n[BACKWARD] Propagating gradient for target=%d...\n", target_token);
	transformer_backward_step(&t, target_token, pos - 1);
	
	/* 5. Apply gradients */
	printf("[UPDATE] Applying gradients (LR=%.3f)...\n", t.nested_lr);
	backward_apply_grads(&t, t.nested_lr);

	/* 6. Forward pass again to measure improvement */
	printf("\n[FORWARD PASS 2] After gradient update...\n");
	
	/* CRITICAL: Reset KV cache for fresh forward pass */
	/* Without this, we'd be appending to positions that already exist */
	for (int l = 0; l < t.config.n_layers; l++)
	{
		t.state.kv_cache[l].current_seq_len = 0;
	}
	
	for (pos = 0; pos < n_tokens; pos++)
	{
		logits = transformer_forward(&t, tokens[pos], pos);
	}
	
	loss_post = compute_ce_loss(logits, t.config.vocab_size, target_token);
	printf("[LOSS] Post-update: %.4f (P(42)=%.2e)\n", 
		loss_post, expf(-loss_post));

	/* 7. Verdict */
	printf("\n═══════════════════════════════════════════════════════════════\n");
	float delta = loss_initial - loss_post;
	if (delta > 0.001f)
	{
		printf("  ✅ SUCCESS: Loss decreased by %.4f\n", delta);
		printf("     The model is LEARNING! Gradient flow is intact.\n");
	}
	else if (delta > 0.0f)
	{
		printf("  ⚠️  MARGINAL: Loss decreased by %.6f (very small)\n", delta);
		printf("     Try increasing LR or running more steps.\n");
	}
	else
	{
		printf("  ❌ FAILURE: Loss did NOT decrease (delta=%.4f)\n", delta);
		printf("     Possible causes:\n");
		printf("     - Gradient flow broken (check backward_apply_grads)\n");
		printf("     - Weights not actually updating (check adapter pointers)\n");
		printf("     - Already at local minimum (unlikely for random target)\n");
	}
	printf("═══════════════════════════════════════════════════════════════\n");

	/* Cleanup */
	free(tokens);
	transformer_free(&t);
	
	return (delta > 0.0f) ? 0 : 1;
}
