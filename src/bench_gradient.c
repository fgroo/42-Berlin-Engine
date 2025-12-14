/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bench_gradient.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 19:00:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 19:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** GRADIENT DIAGNOSTIC BENCHMARK
** =============================
** Instruments the backward pass to understand:
** 1. Gradient magnitude per layer during prefill
** 2. Weight delta after each token
** 3. Loss trajectory over prompt
** 4. FP32 accumulator behavior
**
** This is for debugging the "Paris Loop" issue:
** - If gradients explode, clipping is insufficient
** - If weights change too much, LR is too high
** - If FP32 accumulator works, loss should decrease
*/

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "memory/arena.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <locale.h>

#define TEST_PROMPT "1 + 1 ="

/* Compute L2 norm of FP32 array */
static float compute_norm_f32(const float *arr, int n)
{
	float sum = 0.0f;
	int i = 0;
	while (i < n)
	{
		sum += arr[i] * arr[i];
		i++;
	}
	return sqrtf(sum);
}

/* Compute L2 norm of BF16 array */
static float compute_norm_bf16(const t_bf16 *arr, int n)
{
	float sum = 0.0f;
	int i = 0;
	while (i < n)
	{
		float val = bf16_to_float(arr[i]);
		sum += val * val;
		i++;
	}
	return sqrtf(sum);
}

/* Compute max absolute value */
static float compute_max_abs_f32(const float *arr, int n)
{
	float max = 0.0f;
	int i = 0;
	while (i < n)
	{
		float val = fabsf(arr[i]);
		if (val > max)
			max = val;
		i++;
	}
	return max;
}

/* Print gradient statistics for all layers */
static void print_gradient_stats(t_transformer *t, int token_idx)
{
	int layer;
	int size;
	float grad_x_norm = 0.0f;
	float grad_x_max = 0.0f;
	
	/* grad_x stats */
	if (t->state.grad_x)
	{
		grad_x_norm = compute_norm_f32(t->state.grad_x, t->config.dim);
		grad_x_max = compute_max_abs_f32(t->state.grad_x, t->config.dim);
	}
	
	printf("  [Token %2d] grad_x: norm=%.4e max=%.4e", 
		token_idx, grad_x_norm, grad_x_max);
	
	/* Sample layers: first, middle, last trainable */
	int sample_layers[] = {22, 23, 24, 25};
	int n_samples = 4;
	
	for (int s = 0; s < n_samples; s++)
	{
		layer = sample_layers[s];
		if (layer >= t->config.n_layers)
			continue;
		
		t_fluid_layer *fl = &t->fluid_layers[layer];
		if (!fl->grad_acc)
			continue;
		
		size = fl->w2_weight->size;
		float acc_norm = compute_norm_f32(fl->grad_acc, size);
		float weight_norm = compute_norm_bf16((t_bf16 *)fl->w2_weight->data, size);
		
		if (s == 0)
			printf("\n             L%d: acc=%.4e w=%.4e", layer, acc_norm, weight_norm);
		else
			printf(" | L%d: acc=%.4e", layer, acc_norm);
	}
	printf("\n");
}

/* Run single token forward + backward with diagnostics */
static float run_token_with_diag(t_transformer *t, int token, int target, int pos)
{
	float *logits;
	float max_logit;
	float sum_exp;
	float target_prob;
	float loss;
	int i;
	
	/* Forward pass */
	logits = transformer_forward(t, token, pos);
	
	/* Compute loss manually (same as backward_step) */
	max_logit = logits[0];
	for (i = 1; i < t->config.vocab_size; i++)
		if (logits[i] > max_logit)
			max_logit = logits[i];
	
	sum_exp = 0.0f;
	for (i = 0; i < t->config.vocab_size; i++)
		sum_exp += expf(logits[i] - max_logit);
	
	target_prob = expf(logits[target] - max_logit) / sum_exp;
	loss = -logf(target_prob + 1e-8f);
	
	/* Backward pass (accumulates gradients) */
	transformer_backward_step(t, target, pos);
	
	return loss;
}

int main(int argc, char **argv)
{
	t_transformer t;
	t_tokenizer tok;
	char tokenizer_path[1024];
	char *p;
	int *tokens;
	int n_tokens;
	int i;
	
	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model.safetensors> <config.json>\n", argv[0]);
		return 1;
	}
	setlocale(LC_ALL, "");
	
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║         GRADIENT DIAGNOSTIC BENCHMARK                        ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");
	
	/* Init */
	if (transformer_init(&t, argv[1], argv[2]) != 0)
	{
		fprintf(stderr, "Failed to init transformer\n");
		return 1;
	}
	
	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", argv[2]);
	p = strrchr(tokenizer_path, '/');
	if (p)
		strcpy(p + 1, "tokenizer.json");
	else
		strcpy(tokenizer_path, "tokenizer.json");
	
	if (tokenizer_init(&tok, tokenizer_path) != 0)
	{
		fprintf(stderr, "Failed to init tokenizer\n");
		return 1;
	}
	
	printf("Test Prompt: \"%s\"\n", TEST_PROMPT);
	printf("Config: lr=%.6f, grad_clip=%.2f, frozen_layers=%d\n\n",
		t.nested_lr, GRADIENT_CLIP, FROZEN_LAYERS);
	
	/* Enable learning */
	t.nested_learning = 1;
	t.persistent_mode = 1;
	
	/* Tokenize */
	n_tokens = tokenizer_encode(&tok, TEST_PROMPT, &tokens);
	if (n_tokens < 0)
	{
		fprintf(stderr, "Tokenization failed\n");
		return 1;
	}
	
	printf("Tokens (%d): ", n_tokens);
	for (i = 0; i < n_tokens && i < 10; i++)
		printf("%d ", tokens[i]);
	if (n_tokens > 10)
		printf("...");
	printf("\n\n");
	
	/* Zero grad accumulators before starting */
	printf("Zeroing FP32 gradient accumulators...\n");
	backward_zero_grads(&t);
	
	printf("\n=== PREFILL WITH LEARNING ===\n");
	printf("Showing gradient stats after each backward step:\n\n");
	
	float total_loss = 0.0f;
	int n_learning_steps = 0;
	
	for (i = 0; i < n_tokens - 1; i++)
	{
		int token = tokens[i];
		int target = tokens[i + 1];
		
		printf("Token[%d]=%d -> Target[%d]=%d\n", i, token, i + 1, target);
		
		/* Run with diagnostics */
		float loss = run_token_with_diag(&t, token, target, i);
		
		/* Print gradient stats */
		print_gradient_stats(&t, i);
		
		printf("  Loss=%.4f P(target)=%.2f%%\n\n", 
			loss, 100.0f * expf(-loss));
		
		total_loss += loss;
		n_learning_steps++;
	}
	
	printf("=== PREFILL SUMMARY ===\n");
	printf("Total tokens: %d\n", n_tokens);
	printf("Learning steps: %d\n", n_learning_steps);
	printf("Average loss: %.4f\n", total_loss / n_learning_steps);
	
	/* Check FP32 accumulator norms after prefill */
	printf("\n=== FP32 ACCUMULATOR STATE ===\n");
	for (int layer = 22; layer < t.config.n_layers; layer++)
	{
		t_fluid_layer *fl = &t.fluid_layers[layer];
		if (!fl->grad_acc)
			continue;
		
		int size = fl->w2_weight->size;
		float acc_norm = compute_norm_f32(fl->grad_acc, size);
		float acc_max = compute_max_abs_f32(fl->grad_acc, size);
		float weight_norm = compute_norm_bf16((t_bf16 *)fl->w2_weight->data, size);
		
		printf("Layer %2d: acc_norm=%.4e acc_max=%.4e weight_norm=%.4e\n",
			layer, acc_norm, acc_max, weight_norm);
	}
	
	/* Apply gradients */
	printf("\n=== APPLYING GRADIENTS ===\n");
	printf("Learning rate: %.6f\n", t.nested_lr);
	
	/* Snapshot weights before */
	float weight_norm_before = 0.0f;
	for (int layer = 22; layer < t.config.n_layers; layer++)
	{
		t_bf16 *w = (t_bf16 *)t.fluid_layers[layer].w2_weight->data;
		int size = t.fluid_layers[layer].w2_weight->size;
		weight_norm_before += compute_norm_bf16(w, size);
	}
	
	backward_apply_grads(&t, t.nested_lr);
	
	/* Snapshot weights after */
	float weight_norm_after = 0.0f;
	for (int layer = 22; layer < t.config.n_layers; layer++)
	{
		t_bf16 *w = (t_bf16 *)t.fluid_layers[layer].w2_weight->data;
		int size = t.fluid_layers[layer].w2_weight->size;
		weight_norm_after += compute_norm_bf16(w, size);
	}
	
	printf("Weight norm: before=%.4e after=%.4e delta=%.4e\n",
		weight_norm_before, weight_norm_after, 
		weight_norm_after - weight_norm_before);
	
	/* Now run verification - generate after the equals sign */
	printf("\n=== GENERATION TEST ===\n");
	
	/* Reset KV cache */
	for (i = 0; i < t.config.n_layers; i++)
		t.state.kv_cache[i].current_seq_len = 0;
	
	t.nested_learning = 0; /* Disable learning for generation */
	
	/* Feed prompt again */
	for (i = 0; i < n_tokens; i++)
		transformer_forward(&t, tokens[i], i);
	
	printf("Generating: ");
	int pos = n_tokens;
	for (i = 0; i < 5; i++)
	{
		/* Greedy decode */
		int best = 0;
		float best_val = t.state.logits[0];
		for (int v = 1; v < t.config.vocab_size; v++)
		{
			if (t.state.logits[v] > best_val)
			{
				best_val = t.state.logits[v];
				best = v;
			}
		}
		
		const char *piece = tokenizer_decode(&tok, best);
		printf("%s", piece);
		fflush(stdout);
		
		if (best == 2) /* EOS */
			break;
		
		transformer_forward(&t, best, pos);
		pos++;
	}
	printf("\n");
	
	/* Cleanup */
	free(tokens);
	tokenizer_free(&tok);
	transformer_free(&t);
	
	printf("\n=== DIAGNOSTIC COMPLETE ===\n");
	return 0;
}
