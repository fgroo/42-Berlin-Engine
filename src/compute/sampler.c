/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   sampler.c                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "sampler.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
** Thread-local RNG state for OpenMP thread safety.
** Each thread maintains its own initialized state, eliminating race conditions.
*/
static __thread int		g_rng_init = 0;
static __thread unsigned int	g_rng_seed;

float	sampler_random_float(void)
{
	if (!g_rng_init)
	{
		/* Seed with time + thread-unique value (address of thread-local) */
		g_rng_seed = (unsigned int)time(NULL) ^ (unsigned int)(size_t)&g_rng_init;
		g_rng_init = 1;
	}
	g_rng_seed = g_rng_seed * 1103515245 + 12345;
	return ((float)((g_rng_seed >> 16) & 0x7FFF) / 32767.0f);
}

static int	argmax_loop_f32(float *data, size_t n)
{
	size_t	best_idx;
	float	best_val;
	size_t	i;

	best_idx = 0;
	best_val = -1e9f;
	i = 0;
	while (i < n)
	{
		if (data[i] > best_val)
		{
			best_val = data[i];
			best_idx = i;
		}
		i++;
	}
	return ((int)best_idx);
}

static int	argmax_loop_bf16(t_bf16 *data, size_t n)
{
	size_t	best_idx;
	float	best_val;
	float	val;
	size_t	i;

	best_idx = 0;
	best_val = -1e9f;
	i = 0;
	while (i < n)
	{
		val = bf16_to_float(data[i]);
		if (val > best_val)
		{
			best_val = val;
			best_idx = i;
		}
		i++;
	}
	return ((int)best_idx);
}

int	sample_argmax(const t_tensor *logits)
{
	if (logits->dtype == DTYPE_F32)
		return (argmax_loop_f32((float *)logits->data, logits->size));
	return (argmax_loop_bf16((t_bf16 *)logits->data, logits->size));
}
