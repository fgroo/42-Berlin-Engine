/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   backward.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 18:30:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 18:30:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "inference/inference.h"
#include "config.h"
#include "compute/simd_kernels.h"
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
	int		layer;
	int		i;
	size_t	size;
	float	*grad;
	t_bf16	*weight;
	float	g;
	float	w;
	float	grad_norm;
	float	scale;

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
		/* GLOBAL GRADIENT NORM CLIPPING */
		/* Compute L2 norm of entire gradient vector for this layer */
		grad_norm = 0.0f;
		i = 0;
		while (i < (int)size)
		{
			grad_norm += grad[i] * grad[i];
			i++;
		}
		grad_norm = sqrtf(grad_norm);
		/* Scale gradients if norm exceeds threshold */
		/* max_norm = GRADIENT_CLIP * sqrt(size) for layer-wise scaling */
		/* But simpler: use fixed max_norm = 1.0 for stability */
		scale = 1.0f;
		if (grad_norm > 1.0f)
			scale = 1.0f / grad_norm;
		/* Apply gradient update with scaling */
		i = 0;
		while (i < (int)size)
		{
			g = grad[i] * scale;
			/* Additional per-element clipping as safety net */
			if (g > GRADIENT_CLIP)
				g = GRADIENT_CLIP;
			if (g < -GRADIENT_CLIP)
				g = -GRADIENT_CLIP;
			/* SGD update: w = w - lr * g */
			w = bf16_to_float(weight[i]) - lr * g;
			weight[i] = float_to_bf16(w);
			i++;
		}
		layer--;
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

	c = &t->config;
	fl = &t->fluid_layers[layer];
	grad = fl->grad_acc;
	hb = fl->hb_cache;
	if (!grad || !hb)
		return ;
	#pragma omp parallel for schedule(static) private(j, grad_xi)
	for (i = 0; i < c->dim; i++)
	{
		grad_xi = t->state.grad_x[i];
		/* Gradient clipping before accumulation */
		if (grad_xi > GRADIENT_CLIP)
			grad_xi = GRADIENT_CLIP;
		if (grad_xi < -GRADIENT_CLIP)
			grad_xi = -GRADIENT_CLIP;
		j = 0;
		while (j < c->hidden_dim)
		{
			grad[i * c->hidden_dim + j] += grad_xi * hb[j];
			j++;
		}
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
		t->nl_skipped++;
		if (t->nl_step < 5 || t->nl_step % 20 == 0)
		{
			printf("[NL] Step %d: Loss=%.2f (SKIP - noise) [skipped %d]\n",
				t->nl_step, loss, t->nl_skipped);
			fflush(stdout);
		}
		t->nl_step++;
		return ;
	}
	/* SURPRISE-BASED: Skip low-loss tokens (model already knows) */
	if (loss < LEARNING_THRESHOLD)
	{
		t->nl_skipped++;
		if (t->nl_step < 5 || t->nl_step % 20 == 0)
		{
			printf("[NL] Step %d: Loss=%.2f (SKIP - known) [skipped %d]\n",
				t->nl_step, loss, t->nl_skipped);
			fflush(stdout);
		}
		t->nl_step++;
		return ;
	}
	/* STEP LIMIT: Prevent overfitting */
	if (t->nl_actual_steps >= NL_MAX_STEPS)
	{
		if (t->nl_step < 5 || t->nl_step % 20 == 0)
			printf("[NL] Step %d: Loss=%.2f (SKIP - max steps)\n",
				t->nl_step, loss);
		t->nl_step++;
		return ;
	}
	t->nl_actual_steps++;
	if (t->nl_step < 5 || t->nl_step % 20 == 0)
	{
		printf("[NL] Step %d: Loss=%.2f, P(target)=%.1f%% [LEARN]\n",
			t->nl_step, loss, target_prob * 100);
		fflush(stdout);
	}
	t->nl_step++;
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
