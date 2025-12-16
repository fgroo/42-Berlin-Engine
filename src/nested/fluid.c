/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fluid.h"
#include "compute/ops.h"
#include "../config.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

void	optimizer_sgd(t_fluid_param *param, float lr)
{
	int		size;
	int		i;
	t_bf16	*w;
	t_bf16	*g;
	float	grad_norm;
	float	scale;

	size = param->weight->size;
	w = (t_bf16 *)param->weight->data;
	g = (t_bf16 *)param->grad->data;
	
	/* Compute gradient L2 norm for clipping */
	grad_norm = 0.0f;
	i = 0;
	while (i < size)
	{
		float gv = bf16_to_float(g[i]);
		grad_norm += gv * gv;
		i++;
	}
	grad_norm = sqrtf(grad_norm + 1e-8f);
	
	/* Clip gradient norm to prevent exploding gradients */
	scale = 1.0f;
	if (grad_norm > GRADIENT_NORM_CLIP)
		scale = GRADIENT_NORM_CLIP / grad_norm;
	
	/* Apply gradient update with clipping */
	i = 0;
	while (i < size)
	{
		float gv = bf16_to_float(g[i]) * scale;
		/* Per-element clip as safety */
		if (gv > GRADIENT_CLIP) gv = GRADIENT_CLIP;
		if (gv < -GRADIENT_CLIP) gv = -GRADIENT_CLIP;
		w[i] = float_to_bf16(bf16_to_float(w[i]) - lr * gv);
		i++;
	}
}

static void	sgd_momentum_step(t_bf16 *w, t_bf16 *g, t_bf16 *vel, float *p)
{
	float	v;

	v = p[0] * bf16_to_float(*vel) - p[1] * bf16_to_float(*g);
	*w = float_to_bf16(bf16_to_float(*w) + v);
	*vel = float_to_bf16(v);
}

void	optimizer_sgd_momentum(t_fluid_param_momentum *param,
			float lr, float momentum)
{
	int		size;
	int		i;
	float	p[2];

	size = param->weight->size;
	p[0] = momentum;
	p[1] = lr;
	i = 0;
	while (i < size)
	{
		sgd_momentum_step(
			(t_bf16 *)param->weight->data + i,
			(t_bf16 *)param->grad->data + i,
			(t_bf16 *)param->velocity->data + i, p);
		i++;
	}
}

void	zero_grad(t_fluid_param *param)
{
	memset(param->grad->data, 0, param->grad->size * sizeof(t_bf16));
}

/*
** Clip gradient L2 norm to max_norm to prevent explosion
*/
void	fluid_clip_grad(t_fluid_param *param, float max_norm)
{
	int		size;
	int		i;
	t_bf16	*g;
	float	grad_norm;
	float	scale;

	if (!param || !param->grad || !param->grad->data)
		return;
	size = param->grad->size;
	g = (t_bf16 *)param->grad->data;
	/* Compute L2 norm */
	grad_norm = 0.0f;
	i = 0;
	while (i < size)
	{
		float gv = bf16_to_float(g[i]);
		grad_norm += gv * gv;
		i++;
	}
	grad_norm = sqrtf(grad_norm + 1e-8f);
	/* Scale down if exceeds max_norm */
	if (grad_norm > max_norm)
	{
		scale = max_norm / grad_norm;
		i = 0;
		while (i < size)
		{
			float gv = bf16_to_float(g[i]) * scale;
			g[i] = float_to_bf16(gv);
			i++;
		}
	}
}

/*
** Reset fluid weights to zero for transient learning
** Called after each turn to prevent weight accumulation
*/
void	fluid_reset(t_fluid_param *param)
{
	if (!param || !param->weight || !param->weight->data)
		return;
	memset(param->weight->data, 0, param->weight->size * sizeof(t_bf16));
}
