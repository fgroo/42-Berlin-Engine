/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   sampler.c                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/16 21:15:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "sampler.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>  /* AVX2 SIMD */

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

/*
** AVX2 SIMD argmax for F32 (~8x faster than scalar for 131K vocab)
** Uses parallel max reduction with index tracking via blendv.
*/
static int	argmax_simd_f32(float *data, size_t n)
{
	__m256		vmax;
	__m256		vval;
	__m256		vcmp;
	__m256i		vidx;
	__m256i		vcur;
	__m256i		vstep;
	float		temp_max[8];
	int			temp_idx[8];
	float		best_val;
	int			best_idx;
	size_t		i;
	int			j;

	if (n < 8)
	{
		/* Scalar fallback for tiny arrays */
		best_idx = 0;
		best_val = data[0];
		i = 1;
		while (i < n)
		{
			if (data[i] > best_val)
			{
				best_val = data[i];
				best_idx = (int)i;
			}
			i++;
		}
		return (best_idx);
	}

	/* Initialize: max = first 8 elements, indices = 0..7 */
	vmax = _mm256_loadu_ps(data);
	vidx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	vcur = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
	vstep = _mm256_set1_epi32(8);
	i = 8;

	/* Main SIMD loop: compare 8 floats at a time */
	while (i + 7 < n)
	{
		vval = _mm256_loadu_ps(data + i);
		vcmp = _mm256_cmp_ps(vval, vmax, _CMP_GT_OQ);
		vmax = _mm256_blendv_ps(vmax, vval, vcmp);
		vidx = _mm256_blendv_epi8(vidx, vcur, _mm256_castps_si256(vcmp));
		vcur = _mm256_add_epi32(vcur, vstep);
		i += 8;
	}

	/* Horizontal reduction: find max among 8 lanes */
	_mm256_storeu_ps(temp_max, vmax);
	_mm256_storeu_si256((__m256i *)temp_idx, vidx);
	best_val = temp_max[0];
	best_idx = temp_idx[0];
	j = 1;
	while (j < 8)
	{
		if (temp_max[j] > best_val)
		{
			best_val = temp_max[j];
			best_idx = temp_idx[j];
		}
		j++;
	}

	/* Scalar tail for remainder */
	while (i < n)
	{
		if (data[i] > best_val)
		{
			best_val = data[i];
			best_idx = (int)i;
		}
		i++;
	}
	return (best_idx);
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
		return (argmax_simd_f32((float *)logits->data, logits->size));
	return (argmax_loop_bf16((t_bf16 *)logits->data, logits->size));
}

