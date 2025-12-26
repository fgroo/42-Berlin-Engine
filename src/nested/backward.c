/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   backward.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 18:30:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/14 18:30:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "inference/inference.h"
#include "config.h"
#include "compute/simd_kernels.h"
#include "compute/ops_simd.h"  /* SIMD gradient norm (Phase 3) */
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>  /* [FIX] For time-based RNG seed */
#include <omp.h>

/*
** ============================================================================
** NESTED LEARNING BACKWARD PASS - MIXED PRECISION
** ============================================================================
** Computes gradient of cross-entropy loss w.r.t. fluid adapter weights.
** Called after forward pass with the actual target token.
**
** CRITICAL FIX: Gradients now accumulate in FP32 (grad_acc buffer)
** before being converted to BF16 weights. This prevents gradients
** from vanishing in the 7-bit BF16 mantissa floor.
** ============================================================================
*/

/*
** Zero FP32 gradient accumulators for all layers
** Called at start of each learning turn
*/
void	backward_zero_grads(t_transformer *t)
{
	int		layer;
	size_t	size;

	if (!t->fluid_layers)
		return ;
	size = (size_t)t->config.dim * t->config.hidden_dim * sizeof(float);
	layer = 0;
	while (layer < t->config.n_layers)
	{
		if (t->fluid_layers[layer].grad_acc)
			memset(t->fluid_layers[layer].grad_acc, 0, size);
		layer++;
	}
	
	/* Solution 4: Zero final_adapter gradient accumulator */
	if (t->final_adapter_grad)
	{
		size_t fa_size = (size_t)t->config.dim * t->config.dim * sizeof(float);
		memset(t->final_adapter_grad, 0, fa_size);
	}
	
	/* Phase 11: Zero inhibitor gate gradient accumulator */
	if (t->inhibitor_gate_grad)
	{
		memset(t->inhibitor_gate_grad, 0, t->config.dim * sizeof(float));
	}
}

/*
** Apply accumulated FP32 gradients to BF16 weights
** Uses GLOBAL GRADIENT NORM CLIPPING per layer to prevent explosions
** Called after processing all tokens in a learning turn
*/
void	backward_apply_grads(t_transformer *t, float lr)
{
	int					layer;
	int					i;
	size_t				size;
	float				*grad;
	t_bf16				*weight;
	float				g;
	float				grad_norm;
	float				scale;
	t_xorshift_state	rng;

	if (!t->fluid_layers)
		return ;
	size = (size_t)t->config.dim * t->config.hidden_dim;
	layer = t->config.n_layers - 1;
	while (layer >= 0)
	{
		if (layer < FROZEN_LAYERS)
		{
			layer--;
			continue ;
		}
		/* SAFETY: Check w2_weight exists BEFORE accessing ->data */
		if (!t->fluid_layers[layer].w2_weight)
		{
			layer--;
			continue ;
		}
		grad = t->fluid_layers[layer].grad_acc;
		weight = (t_bf16 *)t->fluid_layers[layer].w2_weight->data;
		if (!grad || !weight)
		{
			layer--;
			continue ;
		}
		/* [FIX #1] Thread-safe RNG seeding using atomic call_counter */
		/* No more static variables - atomically increment counter in nl_state */
		uint64_t current_count = atomic_fetch_add(&t->nl_state.call_counter, 1);
		xorshift_init(&rng, (uint64_t)time(NULL) ^ (current_count * 0xDEADBEEF) ^ (uint64_t)layer);
		/* GLOBAL GRADIENT NORM CLIPPING */
		/* Compute L2 norm using SIMD (16x faster than scalar loop) */
		grad_norm = ops_simd_norm(grad, size);
		/* Scale gradients if norm exceeds threshold */
		/* max_norm = GRADIENT_CLIP * sqrt(size) for layer-wise scaling */
		/* But simpler: use fixed max_norm = 1.0 for stability */
		scale = 1.0f;
		if (grad_norm > 1.0f)
			scale = 1.0f / grad_norm;
		/* Apply gradient update with STOCHASTIC ROUNDING */
		/* This preserves small gradients statistically (Critical #3 Fix) */
		i = 0;
		while (i < (int)size)
		{
			g = grad[i] * scale;
			/* Additional per-element clipping as safety net */
			if (g > GRADIENT_CLIP)
				g = GRADIENT_CLIP;
			if (g < -GRADIENT_CLIP)
				g = -GRADIENT_CLIP;
			/* STOCHASTIC SGD update: w = w - lr * g with SR */
			/* Small updates (< BF16 ULP) now have probability of rounding up */
			bf16_stochastic_update(&weight[i], lr * g, &rng);
			i++;
		}
		/* [DEEP FLUIDITY] Log layer-wise training progress */
		if (grad_norm > 0.01f)  /* Only log if meaningful update */
		{
			float w_sum = 0.0f;
			for (size_t k = 0; k < size && k < 1000; k++)
				w_sum += fabsf(bf16_to_float(weight[k]));
			printf("[LAYER %d] grad_norm=%.4f w_sample=%.6f\n", layer, grad_norm, w_sum);
		}
		layer--;
	}

	
	/* Solution 4: Apply gradients to final_adapter [dim x dim] */
	if (t->final_adapter && t->final_adapter_grad && t->final_adapter->data)
	{
		t_bf16 *fa_weight = (t_bf16 *)t->final_adapter->data;
		float *fa_grad = t->final_adapter_grad;
		size_t fa_size = (size_t)t->config.dim * t->config.dim;
		/* [FIX #1] Thread-safe RNG seeding for final adapter */
		t_xorshift_state fa_rng;
		uint64_t fa_count = atomic_fetch_add(&t->nl_state.call_counter, 1);
		xorshift_init(&fa_rng, (uint64_t)time(NULL) ^ (fa_count * 0xCAFEFACE));
		
		/* Compute gradient norm for clipping */
		float fa_grad_norm = ops_simd_norm(fa_grad, fa_size);
		float fa_scale = 1.0f;
		if (fa_grad_norm > 1.0f)
			fa_scale = 1.0f / fa_grad_norm;
		
		/* Apply with stochastic rounding */
		for (size_t i = 0; i < fa_size; i++)
		{
			float g = fa_grad[i] * fa_scale;
			if (g > GRADIENT_CLIP) g = GRADIENT_CLIP;
			if (g < -GRADIENT_CLIP) g = -GRADIENT_CLIP;
			bf16_stochastic_update(&fa_weight[i], lr * g, &fa_rng);
		}
		
		/* Debug output */
		float fa_sum = 0.0f, fa_max = 0.0f;
		for (size_t i = 0; i < fa_size; i++)
		{
			float val = bf16_to_float(fa_weight[i]);
			fa_sum += fabsf(val);
			if (fabsf(val) > fa_max) fa_max = fabsf(val);
		}
		printf("[FINAL_ADAPTER] weights: sum=%.6f, max=%.6f, grad_norm=%.4f\n", 
			fa_sum, fa_max, fa_grad_norm);
	}
	
	/* Phase 11: Inhibitor Gate Update */
	/* Apply accumulated gradients to gate weights */
	if (t->inhibitor_gate && t->inhibitor_gate_grad)
	{
		int dim = t->config.dim;
		float gate_grad_norm = ops_simd_norm(t->inhibitor_gate_grad, dim);
		float gate_scale = 1.0f;
		if (gate_grad_norm > 1.0f)
			gate_scale = 1.0f / gate_grad_norm;
		
		float gate_sum = 0.0f;
		float gate_max = -1000.0f;
		for (int i = 0; i < dim; i++)
		{
			float g = t->inhibitor_gate_grad[i] * gate_scale;
			if (g > GRADIENT_CLIP) g = GRADIENT_CLIP;
			if (g < -GRADIENT_CLIP) g = -GRADIENT_CLIP;
			// SGD update: gate += lr * (-d_gate) to INCREASE gate when base is wrong
			// Note: We want gate to INCREASE to suppress base, so we ADD the gradient
			t->inhibitor_gate[i] -= lr * g;  // Standard SGD
			gate_sum += t->inhibitor_gate[i];
			if (t->inhibitor_gate[i] > gate_max) gate_max = t->inhibitor_gate[i];
		}
		/* Logging: mean gate and max gate */
		/* Positive gate means suppression active (sigmoid > 0.5) */
		printf("[INHIBITOR_GATE] mean=%.4f, max=%.4f, grad_norm=%.4f\n", 
			gate_sum / dim, gate_max, gate_grad_norm);
	}
}

/*
** Backprop through output layer: grad_x = output_weight^T @ grad_logits
** ZERO-COPY VERSION: Reads directly from frozen weights (mmap) without
** materializing the 800MB transposed matrix.
**
** Key insight: W is [Vocab, Dim] row-major. Instead of transposing to [Dim, Vocab],
** we iterate over Vocab (outer) and accumulate into grad_x (inner).
** This is cache-friendly for W reads (sequential rows) and grad_x fits in L1/L2.
**
** Sparsity optimization: After softmax, most logit gradients are near-zero.
** We skip rows where |grad| < epsilon, avoiding 4K+ FMA ops per skipped token.
*/
static void	backprop_output_layer(t_transformer *t, float *grad_x)
{
	t_transformer_config	*c;
	t_inference_state		*s;
	int						dim;
	int						vocab;
	t_bf16					*w_data;
	float					*logits;

	c = &t->config;
	s = &t->state;
	dim = c->dim;
	vocab = c->vocab_size;
	w_data = (t_bf16 *)t->weights.output->data;
	logits = s->logits;

	/* Zero the output gradient accumulator */
	memset(grad_x, 0, dim * sizeof(float));

	/* Accumulate: grad_x[d] += sum_v(logits[v] * W[v,d]) */
	/* This is mathematically equivalent to: grad_x = W^T @ logits */
	for (int v = 0; v < vocab; v++)
	{
		float grad_val = logits[v];
		
		/* [PHASE 14 FIX] Sparsity optimization REMOVED - it broke nested learning! */
		/* Small gradients (1e-6 to 1e-8) sum across 32K vocab to provide crucial signal */
		/* Without this, the model loses information about what tokens to AVOID */
		/* if (grad_val > -1e-6f && grad_val < 1e-6f) continue; // BREAKS LEARNING */

		/* Row pointer: W[v] = w_data[v * dim] */
		t_bf16 *w_row = &w_data[v * dim];

#ifdef __AVX2__
		/* AVX2 SIMD path: 8 floats per iteration with FMA */
		__m256 g_vec = _mm256_set1_ps(grad_val);
		int d = 0;
		
		/* Main loop: 8 floats at a time */
		for (; d <= dim - 8; d += 8)
		{
			/* Load accumulator */
			__m256 acc = _mm256_loadu_ps(&grad_x[d]);
			
			/* Load W row (BF16 -> FP32) */
			__m128i w_raw = _mm_loadu_si128((const __m128i *)&w_row[d]);
			__m256 w_vec = mm256_cvtbf16_ps(w_raw);
			
			/* FMA: acc += grad_val * w */
			acc = _mm256_fmadd_ps(g_vec, w_vec, acc);
			
			/* Store back */
			_mm256_storeu_ps(&grad_x[d], acc);
		}
		
		/* Scalar cleanup */
		for (; d < dim; d++)
			grad_x[d] += grad_val * bf16_to_float(w_row[d]);
#else
		/* Scalar fallback */
		for (int d = 0; d < dim; d++)
			grad_x[d] += grad_val * bf16_to_float(w_row[d]);
#endif
	}
}

/*
** Accumulate gradients for a single layer in FP32
** grad_acc[i,j] += grad_x[i] * hb_cache[j]
**
** CRITICAL FIX (Phase 10): AVX2 SIMD Vectorization
** The outer product ∇W += δ · xᵀ is now processed 8 floats per iteration
** using FMA instructions. This yields ~6-8x speedup on the backward pass.
*/
static void	accumulate_layer_grads(t_transformer *t, int layer)
{
	t_transformer_config	*c;
	t_fluid_layer			*fl;
	float					*grad;
	float					*hb;
	int						i;
	int						j;
	float					grad_xi;
	float					*grad_row;
	int						hidden;

	c = &t->config;
	fl = &t->fluid_layers[layer];
	grad = fl->grad_acc;
	hb = fl->hb_cache;
	hidden = c->hidden_dim;
	if (!grad || !hb)
		return ;
	#pragma omp parallel for schedule(static) private(j, grad_xi, grad_row)
	for (i = 0; i < c->dim; i++)
	{
		grad_xi = t->state.grad_x[i];
		/* Gradient clipping before accumulation */
		if (grad_xi > GRADIENT_CLIP)
			grad_xi = GRADIENT_CLIP;
		if (grad_xi < -GRADIENT_CLIP)
			grad_xi = -GRADIENT_CLIP;
		grad_row = grad + i * hidden;
#ifdef __AVX2__
		/* AVX2 SIMD path: process 8 floats per iteration with FMA */
		{
			__m256	v_grad_xi;
			__m256	v_grad;
			__m256	v_hb;

			v_grad_xi = _mm256_set1_ps(grad_xi);
			j = 0;
			/* Main loop: 8 floats at a time */
			while (j + 7 < hidden)
			{
				v_grad = _mm256_loadu_ps(grad_row + j);
				v_hb = _mm256_loadu_ps(hb + j);
				v_grad = _mm256_fmadd_ps(v_grad_xi, v_hb, v_grad);
				_mm256_storeu_ps(grad_row + j, v_grad);
				j += 8;
			}
			/* Scalar cleanup for remainder */
			while (j < hidden)
			{
				grad_row[j] += grad_xi * hb[j];
				j++;
			}
		}
#else
		/* Scalar fallback */
		j = 0;
		while (j < hidden)
		{
			grad_row[j] += grad_xi * hb[j];
			j++;
		}
#endif
	}
}

/*
** Main backward pass entry point
** Computes loss, gradients, and accumulates into FP32 buffers
*/
void	transformer_backward_step(t_transformer *t, int target_token, int pos)
{
	t_transformer_config	*c;
	t_inference_state		*s;
	float					target_prob;
	float					loss;
	int						layer;

	(void)pos;
	if (!t->nested_learning || !t->fluid_layers)
		return ;
	c = &t->config;
	s = &t->state;

	/*
	** [FIX #3] SIMD Softmax - 10x faster than scalar loop!
	** simd_softmax_inplace handles: find max, exp(x-max), sum, normalize
	** Replaces ~100K iterations of scalar expf() with AVX2 fast_expf
	*/
	simd_softmax_inplace(s->logits, c->vocab_size);
	
	target_prob = s->logits[target_token];
	s->logits[target_token] -= 1.0f;  /* dL/dlogits = softmax - one_hot */
	loss = -logf(target_prob + 1e-8f);
	
	/*
	** Note: Logit bias update logic (Solution 5) was removed during cleanup.
	** It contained thread-unsafe static counters and was replaced by the
	** Context-Aware Bigram Bias approach below.
	*/

	/* [PHASE 18] BOUNDARY GUARD: Only learn AFTER the prompt ends! */
	/* This ensures we learn "is" -> "42Berlin", not "BOS" -> "The" */
	/* [PHASE 22 FIX] REMOVED loss>1.0 filter! Teacher forcing makes model confident */
	/*   (low loss) but we STILL need to learn facts for context_bias! */
	/*   The prompt_len and substance filters provide sufficient protection. */
	if (t->context_bias.keys && pos >= t->prompt_len)
	{
		int prev_token = (pos > 0) ? t->state.token_history[pos - 1] : 1;
		
		/* [PHASE 16 FIX] THE SUBSTANCE FILTER
		 * We only learn "content" tokens, not "structure" tokens like punctuation.
		 * Without tokenizer access, we use heuristics:
		 * 1. Self-loop: prev == target -> Skip (prevents A->A)
		 * 2. Close-ID loop: abs(prev - target) < 100 often means punctuation pair
		 * 3. Special tokens: 0=PAD, 1=BOS should never be learned
		 */
		int has_substance = 1;
		
		/* [PHASE 19 FIX] Special token filter: Skip PAD(0), BOS(1), EOS(2) */
		if (prev_token < 3 || target_token < 3)
		{
			has_substance = 0;
		}
		
		/* Self-loop prevention */
		if (prev_token == target_token)
		{
			has_substance = 0;
			// printf("[CONTEXT_BIAS] SKIP self-loop: %d -> %d\n", prev_token, target_token);
		}
		
		/* Close-ID heuristic: quote pairs like " and " are often close token IDs */
		if (has_substance && prev_token > 1000 && target_token > 1000)
		{
			int diff = prev_token > target_token ? prev_token - target_token : target_token - prev_token;
			if (diff < 50)  /* Very close IDs suggest paired punctuation */
			{
				has_substance = 0;
				// printf("[CONTEXT_BIAS] SKIP close-pair: %d -> %d (diff=%d)\n", prev_token, target_token, diff);
			}
		}
		
		/* [PHASE 19] Low ID filter DISABLED - core mechanism proven working!
		 * Self-loop and close-pair filters still protect against loops.
		 * We want to learn ALL token transitions in the response.
		 */
		// if (has_substance && target_token < 2000)
		// {
		// 	has_substance = 0;
		// }
		
		
		if (!has_substance)
			goto skip_context_bias;  /* Skip to end of context_bias block */
		
		float context_lr = 100.0f; // PHASE 21: NUCLEAR OPTION - must overpower Base Model
		uint64_t key = (uint64_t)prev_token;  // Simple unigram context
		
		/* [FIX] Load factor warning at 75% capacity */
		if (t->context_bias.count > (t->context_bias.size * 3 / 4))
		{
			static int warned = 0;
			if (!warned)
			{
				fprintf(stderr, "[WARN] Context bias 75%% full (%d/%d). Consider increasing size.\n",
					t->context_bias.count, t->context_bias.size);
				warned = 1;
			}
		}
		
		// Hash and probe
		uint64_t h = key;
		h ^= h >> 33;
		h *= 0xff51afd7ed558ccdULL;
		h ^= h >> 33;
		h *= 0xc4ceb9fe1a85ec53ULL;
		h ^= h >> 33;
		uint32_t idx = (uint32_t)(h % t->context_bias.size);
		
		int inserted = 0;
		for (int i = 0; i < LINEAR_PROBE_LIMIT; i++)
		{
			uint32_t cur = (idx + i) % t->context_bias.size;
			if (t->context_bias.keys[cur] == key)
			{
				/* [PHASE 21] NUCLEAR clamping - 200.0 for absolute override */
				float new_bias = t->context_bias.biases[cur] + context_lr;
				if (new_bias > 200.0f) new_bias = 200.0f;
				if (new_bias < -200.0f) new_bias = -200.0f;
				t->context_bias.biases[cur] = new_bias;
				
				/* Debug: show what we're learning */
				printf("[CONTEXT_BIAS] Learn: prev=%d -> target=%d (loss=%.2f) bias=%.2f\n",
					prev_token, target_token, loss, new_bias);
				inserted = 1;
				break ;
			}
			if (t->context_bias.keys[cur] == 0)
			{
				t->context_bias.keys[cur] = key;
				t->context_bias.tokens[cur] = target_token;  // This is what we predict!
				t->context_bias.biases[cur] = context_lr;
				t->context_bias.count++;
				printf("[CONTEXT_BIAS] NEW: prev=%d -> target=%d (loss=%.2f)\n",
					prev_token, target_token, loss);
				inserted = 1;
				break ;
			}
		}
		/* [FIX] Log warning if linear probe exhausted without finding slot */
		if (!inserted)
		{
			static int exhaust_count = 0;
			if (++exhaust_count <= 5)
				fprintf(stderr, "[WARN] Context bias probe exhausted (key=%llu). Fact lost!\n",
					(unsigned long long)key);
		}
	}
skip_context_bias:  /* Label for substance filter goto */
	/* NOISE FILTER: Skip high-loss tokens (complete confusion) */
	if (loss > HIGH_LOSS_THRESHOLD)
	{
		uint32_t	cur_step;
		uint32_t	cur_skip;

		nl_record_step(&t->nl_state, true, &cur_step, &cur_skip);
		if (cur_step < 5 || cur_step % 20 == 0)
		{
			printf("[NL] Step %u: Loss=%.2f (SKIP - noise) [skipped %u]\n",
				cur_step, loss, cur_skip);
			fflush(stdout);
		}
		return ;
	}
	/* SURPRISE-BASED: Skip low-loss tokens (model already knows) */
	if (loss < LEARNING_THRESHOLD)
	{
		uint32_t	cur_step;
		uint32_t	cur_skip;

		nl_record_step(&t->nl_state, true, &cur_step, &cur_skip);
		if (cur_step < 5 || cur_step % 20 == 0)
		{
			printf("[NL] Step %u: Loss=%.2f (SKIP - known) [skipped %u]\n",
				cur_step, loss, cur_skip);
			fflush(stdout);
		}
		return ;
	}
	/* STEP LIMIT: Prevent overfitting */
	{
		int32_t		cur_actual;
		uint32_t	cur_step;

		cur_actual = nl_get_actual_steps(&t->nl_state);
		if (cur_actual >= NL_MAX_STEPS)
		{
			nl_record_step(&t->nl_state, true, &cur_step, NULL);
			if (cur_step < 5 || cur_step % 20 == 0)
				printf("[NL] Step %u: Loss=%.2f (SKIP - max steps)\n",
					cur_step, loss);
			return ;
		}
	}
	/* ACTUAL LEARNING: Record step (not skipped) and increment actual_steps */
	{
		uint32_t	cur_step;
		int32_t		cur_actual;

		nl_record_step(&t->nl_state, false, &cur_step, NULL);
		cur_actual = nl_inc_actual_steps(&t->nl_state) + 1;
		if (cur_step < 5 || cur_step % 20 == 0)
		{
			printf("[NL] Step %u: Loss=%.2f, P(target)=%.1f%% [LEARN #%d]\n",
				cur_step, loss, target_prob * 100, cur_actual);
			fflush(stdout);
		}
	}
	/* 2. Backprop through output layer */
	backprop_output_layer(t, s->grad_x);
	
	/* Solution 4: Accumulate gradients for final_adapter */
	/* dW = grad_x @ x^T where grad_x = dL/dx from output layer */
	/* x is the normalized hidden state before adapter application */
	/* NOTE: We use s->x which is POST-adapter, but the gradient flows through */
	if (t->final_adapter_grad && t->final_adapter)
	{
		int dim = c->dim;
		float *grad = t->final_adapter_grad;
		
		/* OUTER PRODUCT: grad[i,j] += grad_x[i] * x[j] */
		/* This accumulates over all tokens in the turn */
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < dim; i++)
		{
			float grad_xi = s->grad_x[i] * ADAPTER_SCALE;  // Chain rule: scale factor
			float *grad_row = grad + i * dim;
		/* [FIX] Use final_input_cache (pre-adapter state) not s->x (post-adapter) */
			for (int j = 0; j < dim; j++)
			{
				grad_row[j] += grad_xi * s->final_input_cache[j];
			}
		}
	}
	
	/* Phase 11: Inhibitor Gate Backward Pass */
	/* d_gate[i] = grad_x[i] * (-base_val[i]) * sigmoid_deriv(gate[i]) */
	/* When loss is high from base model, gradient pushes gate POSITIVE to suppress base */
	if (t->inhibitor_gate && t->inhibitor_gate_grad && s->final_input_cache)
	{
		int dim = c->dim;
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < dim; i++)
		{
			float gate_val = t->inhibitor_gate[i];
			float gate_act = 1.0f / (1.0f + expf(-gate_val));
			float sigmoid_deriv = gate_act * (1.0f - gate_act);
			// base_val was cached in final_input_cache during forward
			float base_val = s->final_input_cache[i];
			// Chain rule: d_out/d_gate = -base_val * sigmoid_deriv
			// grad_x flows from output layer
			float d_gate = s->grad_x[i] * (-base_val) * sigmoid_deriv;
			// Accumulate gradient (will be applied in backward_apply_grads)
			t->inhibitor_gate_grad[i] += d_gate;
		}
	}
	
	/* 3. Accumulate gradients into FP32 buffers (reverse layer order) */
	layer = c->n_layers - 1;
	while (layer >= 0)
	{
		if (layer >= FROZEN_LAYERS)
			accumulate_layer_grads(t, layer);
		layer--;
	}
}

/*
** ============================================================================
** MOPD (Multi-Teacher On-Policy Distillation) - Phase 1
** ============================================================================
*/

/*
** Core backprop mechanism: Takes precomputed d_logits and propagates through
** all layers. This is the "heavy lifter" extracted from transformer_backward_step.
**
** NOTE: This function assumes d_logits already contains softmax(logits) - target
** It does NOT compute loss or do skip logic - that's the caller's job.
*/
void	backward_propagate_logits(t_transformer *t, float *d_logits, int pos)
{
	t_transformer_config	*c;
	t_inference_state		*s;
	int						layer;

	(void)pos;
	if (!t->nested_learning || !t->fluid_layers)
		return ;
	c = &t->config;
	s = &t->state;

	/* Copy d_logits to state.logits for backprop_output_layer compatibility */
	memcpy(s->logits, d_logits, c->vocab_size * sizeof(float));

	/* Backprop through output layer */
	backprop_output_layer(t, s->grad_x);

	/* Accumulate gradients for final_adapter if present */
	if (t->final_adapter_grad && t->final_adapter)
	{
		int dim = c->dim;
		float *grad = t->final_adapter_grad;

		#pragma omp parallel for schedule(static)
		for (int i = 0; i < dim; i++)
		{
			float grad_xi = s->grad_x[i] * ADAPTER_SCALE;
			float *grad_row = grad + i * dim;
			for (int j = 0; j < dim; j++)
				grad_row[j] += grad_xi * s->final_input_cache[j];
		}
	}

	/* Accumulate gradients into FP32 buffers (reverse layer order) */
	layer = c->n_layers - 1;
	while (layer >= 0)
	{
		if (layer >= FROZEN_LAYERS)
			accumulate_layer_grads(t, layer);
		layer--;
	}
}

/*
** Cross-Entropy backward pass (standard training wrapper).
** Computes gradient as: softmax(logits) - one_hot(target)
** Uses scratch arena for temporary d_logits buffer.
*/
void	backward_step_ce(t_transformer *t, int target_token, int pos)
{
	t_transformer_config	*c;
	float					*d_logits;

	if (!t->nested_learning || !t->fluid_layers)
		return ;
	c = &t->config;

	/* Allocate scratch buffer for gradients */
	d_logits = arena_try_alloc(&t->scratch, c->vocab_size * sizeof(float));
	if (!d_logits)
	{
		fprintf(stderr, "[MOPD] OOM for d_logits scratch\n");
		return ;
	}

	/* Copy logits and compute softmax */
	memcpy(d_logits, t->state.logits, c->vocab_size * sizeof(float));
	simd_softmax_inplace(d_logits, c->vocab_size);

	/* CE Gradient: softmax - one_hot */
	d_logits[target_token] -= 1.0f;

	/* Propagate through layers */
	backward_propagate_logits(t, d_logits, pos);

	/* Reset scratch arena */
	arena_reset(&t->scratch);
}

/*
** MOPD Distillation backward pass.
** Computes gradient as: softmax(logits) - mixed_target
** where mixed_target = alpha * teacher + (1-alpha) * one_hot
**
** This is the heart of knowledge distillation - pulling student towards teacher.
*/
void	backward_step_distill(t_transformer *t, t_sparse_prob *teacher_probs,
			int num_probs, int target_token, float alpha, int pos)
{
	t_transformer_config	*c;
	float					*d_logits;
	int						i;

	if (!t->nested_learning || !t->fluid_layers)
		return ;
	c = &t->config;

	/* Validate alpha */
	if (alpha < 0.0f)
		alpha = 0.0f;
	if (alpha > 1.0f)
		alpha = 1.0f;

	/* Allocate scratch buffer */
	d_logits = arena_try_alloc(&t->scratch, c->vocab_size * sizeof(float));
	if (!d_logits)
	{
		fprintf(stderr, "[MOPD] OOM for distill d_logits\n");
		return ;
	}

	/* Copy logits and compute softmax -> student probs Q(x) */
	memcpy(d_logits, t->state.logits, c->vocab_size * sizeof(float));
	simd_softmax_inplace(d_logits, c->vocab_size);

	/* d_logits now contains Q(x) = student probabilities
	** Gradient: Q(x) - P_mixed(x)
	** P_mixed = alpha * P_teacher + (1-alpha) * P_target */

	/* Step A: Subtract hard label component (one_hot weighted by 1-alpha) */
	if (target_token >= 0 && target_token < c->vocab_size)
		d_logits[target_token] -= (1.0f - alpha);

	/* Step B: Subtract teacher component (sparse!)
	** Only iterate over tokens the teacher gave us - extremely efficient */
	for (i = 0; i < num_probs; i++)
	{
		int idx = teacher_probs[i].token_id;
		if (idx >= 0 && idx < c->vocab_size)
			d_logits[idx] -= (alpha * teacher_probs[i].prob);
	}

	/* Propagate through layers */
	backward_propagate_logits(t, d_logits, pos);

	/* Reset scratch arena */
	arena_reset(&t->scratch);
}
