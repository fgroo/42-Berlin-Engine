/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   optimizer.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 02:35:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/15 02:35:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fluid.h"
#include "compute/ops.h"
#include "compute/ops_math_fast.h"
#include "../config.h"
#include <string.h>
#include <math.h>
#include <omp.h>

/*
** ============================================================================
** ADAMW OPTIMIZER WITH FP32 ACCUMULATORS
** ============================================================================
** Critical fix for BF16 gradient precision loss.
**
** Problem: BF16 has only 7-bit mantissa. When doing:
**   weight -= learning_rate * gradient
** Small updates (< 2^-7 of weight magnitude) simply vanish.
**
** Solution:
** 1. Keep momentum (m) and velocity (v) in FP32 (not BF16!)
** 2. Accumulate updates in FP32
** 3. Use STOCHASTIC ROUNDING when converting back to BF16
**    - Gives small updates a PROBABILITY of surviving
**    - P(round up) = fractional_part / ulp
** ============================================================================
*/

/*
** AdamW state for a single parameter tensor
** Momentum and velocity are FP32 for precision
*/
typedef struct s_adamw_state
{
	float	*m;			/* First moment (FP32) */
	float	*v;			/* Second moment (FP32) */
	int		t;			/* Timestep counter */
	size_t	size;		/* Number of parameters */
}	t_adamw_state;

/*
** Initialize AdamW state for a parameter tensor
*/
t_adamw_state	*adamw_state_init(size_t size)
{
	t_adamw_state	*state;

	state = malloc(sizeof(t_adamw_state));
	if (!state)
		return (NULL);
	state->m = calloc(size, sizeof(float));
	state->v = calloc(size, sizeof(float));
	if (!state->m || !state->v)
	{
		free(state->m);
		free(state->v);
		free(state);
		return (NULL);
	}
	state->t = 0;
	state->size = size;
	return (state);
}

/*
** Free AdamW state
*/
void	adamw_state_free(t_adamw_state *state)
{
	if (!state)
		return ;
	free(state->m);
	free(state->v);
	free(state);
}

/*
** AdamW step with FP32 accumulators and stochastic rounding
**
** @param weights: BF16 parameter tensor (in/out)
** @param grads: BF16 gradient tensor (in)
** @param state: FP32 momentum/velocity state
** @param lr: Learning rate
** @param beta1: First moment decay (default: 0.9)
** @param beta2: Second moment decay (default: 0.999)
** @param eps: Numerical stability (default: 1e-8)
** @param weight_decay: L2 regularization (default: 0.01)
*/
void	optimizer_adamw_step(
	t_bf16 *weights,
	const t_bf16 *grads,
	t_adamw_state *state,
	float lr,
	float beta1,
	float beta2,
	float eps,
	float weight_decay)
{
	size_t	i;
	float	g;
	float	w;
	float	m_hat;
	float	v_hat;
	float	update;
	float	w_new;
	float	bias_corr1;
	float	bias_corr2;

	state->t++;
	bias_corr1 = 1.0f - powf(beta1, (float)state->t);
	bias_corr2 = 1.0f - powf(beta2, (float)state->t);
	#pragma omp parallel for schedule(static) private(g, w, m_hat, v_hat, update, w_new)
	for (i = 0; i < state->size; i++)
	{
		/* 1. Load BF16 -> FP32 */
		g = bf16_to_float(grads[i]);
		w = bf16_to_float(weights[i]);
		/* 2. Gradient clipping (per-element) */
		if (g > GRADIENT_CLIP)
			g = GRADIENT_CLIP;
		if (g < -GRADIENT_CLIP)
			g = -GRADIENT_CLIP;
		/* 3. Update biased first moment estimate (momentum) */
		state->m[i] = beta1 * state->m[i] + (1.0f - beta1) * g;
		/* 4. Update biased second moment estimate (RMSprop) */
		state->v[i] = beta2 * state->v[i] + (1.0f - beta2) * g * g;
		/* 5. Compute bias-corrected estimates */
		m_hat = state->m[i] / bias_corr1;
		v_hat = state->v[i] / bias_corr2;
		/* 6. Compute update with weight decay (AdamW decouples decay) */
		update = (m_hat / (sqrtf(v_hat) + eps)) + (weight_decay * w);
		/* 7. Apply update in FP32 */
		w_new = w - lr * update;
		/* 8. Convert back to BF16 with STOCHASTIC ROUNDING */
		/* This is critical: small updates now have a chance to survive! */
		weights[i] = fp32_to_bf16_stochastic(w_new);
	}
}

/*
** SGD with Nesterov momentum and stochastic rounding
** Simpler alternative to AdamW, still with FP32 accumulator
**
** @param weights: BF16 parameter tensor (in/out)
** @param grads: BF16 gradient tensor (in)
** @param velocity: FP32 velocity buffer (in/out)
** @param size: Number of parameters
** @param lr: Learning rate
** @param momentum: Momentum coefficient (default: 0.9)
*/
void	optimizer_sgd_nesterov(
	t_bf16 *weights,
	const t_bf16 *grads,
	float *velocity,
	size_t size,
	float lr,
	float momentum)
{
	size_t	i;
	float	g;
	float	w;
	float	v_new;
	float	w_new;

	#pragma omp parallel for schedule(static) private(g, w, v_new, w_new)
	for (i = 0; i < size; i++)
	{
		g = bf16_to_float(grads[i]);
		w = bf16_to_float(weights[i]);
		/* Gradient clipping */
		if (g > GRADIENT_CLIP)
			g = GRADIENT_CLIP;
		if (g < -GRADIENT_CLIP)
			g = -GRADIENT_CLIP;
		/* Nesterov momentum: v = momentum * v + grad */
		v_new = momentum * velocity[i] + g;
		velocity[i] = v_new;
		/* Update: w -= lr * (momentum * v_new + grad) */
		w_new = w - lr * (momentum * v_new + g);
		/* Stochastic rounding */
		weights[i] = fp32_to_bf16_stochastic(w_new);
	}
}

/*
** Simple SGD with stochastic rounding (replaces old optimizer_sgd)
** Uses FP32 intermediate for precision
*/
void	optimizer_sgd_sr(t_bf16 *weights, const t_bf16 *grads,
			size_t size, float lr)
{
	size_t	i;
	float	g;
	float	w;
	float	w_new;

	#pragma omp parallel for schedule(static) private(g, w, w_new)
	for (i = 0; i < size; i++)
	{
		g = bf16_to_float(grads[i]);
		w = bf16_to_float(weights[i]);
		/* Gradient clipping */
		if (g > GRADIENT_CLIP)
			g = GRADIENT_CLIP;
		if (g < -GRADIENT_CLIP)
			g = -GRADIENT_CLIP;
		w_new = w - lr * g;
		/* Stochastic rounding - critical for small updates! */
		weights[i] = fp32_to_bf16_stochastic(w_new);
	}
}
