/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fluid.h"
#include "compute/ops.h"
#include <stdio.h>
#include <string.h>

void	optimizer_sgd(t_fluid_param *param, float lr)
{
	int		size;
	int		i;
	t_bf16	*w;
	t_bf16	*g;

	size = param->weight->size;
	w = (t_bf16 *)param->weight->data;
	g = (t_bf16 *)param->grad->data;
	i = 0;
	while (i < size)
	{
		w[i] = float_to_bf16(bf16_to_float(w[i]) - lr * bf16_to_float(g[i]));
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
