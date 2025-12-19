/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_norm.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/08 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include <math.h>
#include <stdio.h>
#include <immintrin.h>

/*
** AVX2 SIMD RMSNorm - THE FREE SPEED PATCH
** RMSNorm runs 64 times per token (2x per layer × 32 layers)
** Formula: out = x / sqrt(mean(x²) + eps) * weight
** 
** Optimization: Vectorized sum-of-squares + vectorized normalize
** Expected speedup: 4-8x over scalar implementation
*/

static float	compute_rms_f32_avx2(const float *rowp, int dim)
{
	__m256	ss_vec;
	__m256	v;
	__m128	lo;
	__m128	hi;
	float	ss;
	int		j;

	ss_vec = _mm256_setzero_ps();
	j = 0;
	/* AVX2 loop: process 8 floats at a time */
	while (j + 7 < dim)
	{
		v = _mm256_loadu_ps(rowp + j);
		ss_vec = _mm256_fmadd_ps(v, v, ss_vec);
		j += 8;
	}
	/* Horizontal sum: reduce 8 floats to 1 */
	lo = _mm256_castps256_ps128(ss_vec);
	hi = _mm256_extractf128_ps(ss_vec, 1);
	lo = _mm_add_ps(lo, hi);
	lo = _mm_hadd_ps(lo, lo);
	lo = _mm_hadd_ps(lo, lo);
	ss = _mm_cvtss_f32(lo);
	/* Scalar remainder */
	while (j < dim)
	{
		ss += rowp[j] * rowp[j];
		j++;
	}
	return (ss / dim);
}

static void	normalize_row_f32_avx2(float *out_row, const float *in_row,
				const t_bf16 *w_row, int dim, float inv_rms)
{
	__m256	inv_rms_vec;
	__m256	v;
	__m256i	w_32;
	__m128i	bf16_w;
	__m256	w_f32;
	__m256	res;
	int		j;

	inv_rms_vec = _mm256_set1_ps(inv_rms);
	j = 0;
	/* AVX2 loop: normalize and scale 8 floats at a time */
	while (j + 7 < dim)
	{
		/* Load 8 input values */
		v = _mm256_loadu_ps(in_row + j);
		/* Load 8 BF16 weights, convert to F32 */
		bf16_w = _mm_loadu_si128((__m128i *)(w_row + j));
		w_32 = _mm256_cvtepu16_epi32(bf16_w);
		w_32 = _mm256_slli_epi32(w_32, 16);
		w_f32 = _mm256_castsi256_ps(w_32);
		/* out = x * inv_rms * weight */
		res = _mm256_mul_ps(v, inv_rms_vec);
		res = _mm256_mul_ps(res, w_f32);
		_mm256_storeu_ps(out_row + j, res);
		j += 8;
	}
	/* Scalar remainder */
	while (j < dim)
	{
		out_row[j] = in_row[j] * inv_rms * bf16_to_float(w_row[j]);
		j++;
	}
}

static int	get_norm_dim(const t_tensor *x)
{
	if (x->ndim == 2 && x->shape[1] == 1)
		return (x->shape[0]);
	return (x->shape[x->ndim - 1]);
}

static void	rmsnorm_f32_avx2(t_tensor *out, const t_tensor *x,
				const t_tensor *w, float eps)
{
	int		dim;
	int		num_tokens;
	int		i;

	dim = get_norm_dim(x);
	num_tokens = x->size / dim;
	// [HOTFIX] Issue #6: Tokens are independent, parallelize!
	#pragma omp parallel for schedule(static)
	for (i = 0; i < num_tokens; i++)
	{
		float	*in_row = (float *)x->data + i * dim;
		float	*out_row = (float *)out->data + i * dim;
		float	ss = compute_rms_f32_avx2(in_row, dim);
		float	inv_rms = 1.0f / sqrtf(ss + eps);
		normalize_row_f32_avx2(out_row, in_row, (t_bf16 *)w->data, dim, inv_rms);
	}
}

void	op_rmsnorm(t_tensor *out, const t_tensor *x,
			const t_tensor *w, float epsilon)
{
	int	is_f32;

	is_f32 = (x->dtype == DTYPE_F32);
	is_f32 = is_f32 && (out->dtype == DTYPE_F32);
	is_f32 = is_f32 && (w->dtype == DTYPE_BF16);
	if (is_f32)
		rmsnorm_f32_avx2(out, x, w, epsilon);
	else
		fprintf(stderr, "RMSNorm unsupported dtypes\n");
}
