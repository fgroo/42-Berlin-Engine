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
		grad = t->fluid_layers[layer].grad_acc;
		weight = (t_bf16 *)t->fluid_layers[layer].w2_weight->data;
		if (!grad || !weight)
		{
			layer--;
			continue ;
		}
		/* Initialize per-layer RNG with unique seed */
		/* layer + time + 1 ensures non-zero seed for different layers */
		xorshift_init(&rng, (uint64_t)layer * 0xDEADBEEF + 0x42424242ULL);
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
		layer--;
	}
	/* DEBUG: Print weight statistics after gradient application */
	{
		float weight_sum = 0.0f;
		float weight_max = 0.0f;
		int layer_23 = 23;  /* Check layer 23 as sample */
		if (layer_23 < t->config.n_layers && t->fluid_layers[layer_23].w2_weight)
		{
			t_bf16 *w = (t_bf16 *)t->fluid_layers[layer_23].w2_weight->data;
			size_t sz = (size_t)t->config.dim * t->config.hidden_dim;
			for (size_t i = 0; i < sz; i++)
			{
				float val = bf16_to_float(w[i]);
				weight_sum += fabsf(val);
				if (fabsf(val) > weight_max)
					weight_max = fabsf(val);
			}
			printf("[DEBUG] Layer 23 weights: sum=%.6f, max=%.6f\n", weight_sum, weight_max);
		}
	}
}

/*
** Backprop through output layer: grad_x = output_weight^T @ grad_logits
** Uses pre-transposed weights for cache-friendly access
*/
static void	backprop_output_layer(t_transformer *t, float *grad_x)
{
	t_transformer_config	*c;
	t_inference_state		*s;
	int						d;
	int						v;
	int						vocab;
	float					sum;

	c = &t->config;
	s = &t->state;
	vocab = c->vocab_size;
	memset(grad_x, 0, c->dim * sizeof(float));
	if (t->output_weight_T)
	{
		/* FAST PATH: Pre-transposed BF16 weights with SIMD */
		#pragma omp parallel for schedule(static)
		for (d = 0; d < c->dim; d++)
			grad_x[d] = simd_dot_bf16_f32(
				t->output_weight_T + d * vocab, s->logits, vocab);
	}
	else
	{
		/* FALLBACK: Naive column-major access */
		t_bf16 *ow_data = (t_bf16 *)t->weights.output->data;
		d = 0;
		while (d < c->dim)
		{
			sum = 0.0f;
			v = 0;
			while (v < vocab)
			{
				sum += bf16_to_float(ow_data[v * c->dim + d]) * s->logits[v];
				v++;
			}
			grad_x[d] = sum;
			d++;
		}
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
	float					max_logit;
	float					sum_exp;
	float					target_prob;
	float					loss;
	float					inv_sum;
	int						i;
	int						layer;

	(void)pos;
	if (!t->nested_learning || !t->fluid_layers)
		return ;
	c = &t->config;
	s = &t->state;
	/* 1. Compute softmax probabilities from logits */
	max_logit = s->logits[0];
	i = 1;
	while (i < c->vocab_size)
	{
		if (s->logits[i] > max_logit)
			max_logit = s->logits[i];
		i++;
	}
	sum_exp = 0.0f;
	i = 0;
	while (i < c->vocab_size)
	{
		s->logits[i] = expf(s->logits[i] - max_logit);
		sum_exp += s->logits[i];
		i++;
	}
	/* Softmax and cross-entropy gradient in one pass */
	inv_sum = 1.0f / sum_exp;
	i = 0;
	while (i < c->vocab_size)
	{
		s->logits[i] = s->logits[i] * inv_sum;
		i++;
	}
	target_prob = s->logits[target_token];
	s->logits[target_token] -= 1.0f;
	loss = -logf(target_prob + 1e-8f);
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
	/* 3. Accumulate gradients into FP32 buffers (reverse layer order) */
	layer = c->n_layers - 1;
	while (layer >= 0)
	{
		if (layer >= FROZEN_LAYERS)
			accumulate_layer_grads(t, layer);
		layer--;
	}
}
