/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   sampler_temp.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/07 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/07 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "sampler.h"
#include <math.h>

float	sampler_random_float(void);

static float	find_max_f32(float *data, size_t n)
{
	float	max_l;
	size_t	i;

	max_l = -1e9f;
	i = 0;
	while (i < n)
	{
		if (data[i] > max_l)
			max_l = data[i];
		i++;
	}
	return (max_l);
}

static float	find_max_bf16(t_bf16 *data, size_t n)
{
	float	max_l;
	float	val;
	size_t	i;

	max_l = -1e9f;
	i = 0;
	while (i < n)
	{
		val = bf16_to_float(data[i]);
		if (val > max_l)
			max_l = val;
		i++;
	}
	return (max_l);
}

static void	compute_probs(float *p, const t_tensor *l, float *params)
{
	float	sum;
	size_t	i;

	sum = 0.0f;
	i = 0;
	while (i < l->size)
	{
		if (l->dtype == DTYPE_F32)
			p[i] = expf((((float *)l->data)[i] - params[0]) / params[1]);
		else
			p[i] = expf((bf16_to_float(((t_bf16 *)l->data)[i])
						- params[0]) / params[1]);
		sum += p[i];
		i++;
	}
	i = 0;
	while (i < l->size)
	{
		p[i] /= sum;
		i++;
	}
}

static int	sample_from_probs(float *probs, size_t n)
{
	float	r;
	float	cum;
	size_t	idx;

	r = sampler_random_float();
	cum = 0.0f;
	idx = 0;
	while (idx < n)
	{
		cum += probs[idx];
		if (r < cum)
			break ;
		idx++;
	}
	return ((int)idx);
}

int	sample_temperature(const t_tensor *l, float temp, t_arena *scratch)
{
	size_t	saved;
	float	*probs;
	float	params[2];
	int		result;

	saved = scratch->offset;
	probs = arena_alloc(scratch, l->size * sizeof(float));
	if (l->dtype == DTYPE_F32)
		params[0] = find_max_f32((float *)l->data, l->size);
	else
		params[0] = find_max_bf16((t_bf16 *)l->data, l->size);
	params[1] = temp;
	compute_probs(probs, l, params);
	result = sample_from_probs(probs, l->size);
	scratch->offset = saved;
	return (result);
}
