/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   simd_kernels.h                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 18:30:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 18:30:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef SIMD_KERNELS_H
# define SIMD_KERNELS_H

# include <stdint.h>
# include "ops_internal.h"

/*
** ============================================================================
** SIMD KERNEL PRIMITIVES
** ============================================================================
** Reusable AVX2/FMA primitives for:
**   - BF16 <-> F32 conversion
**   - Dot products (BF16 x F32, F32 x F32)
**   - Horizontal sum reduction
**   - Fused scalar-vector operations
**
** All functions are inline for maximum performance.
** Fallback scalar paths provided for non-AVX2 builds.
** ============================================================================
*/

# ifdef __AVX2__
#  include <immintrin.h>
#  define SIMD_ENABLED 1
# else
#  define SIMD_ENABLED 0
# endif

/*
** Horizontal sum of 8 floats in __m256 register
** Returns scalar float
*/
# if SIMD_ENABLED
static inline float	simd_hsum_ps(__m256 v)
{
	__m128	lo;
	__m128	hi;

	lo = _mm256_castps256_ps128(v);
	hi = _mm256_extractf128_ps(v, 1);
	lo = _mm_add_ps(lo, hi);
	lo = _mm_hadd_ps(lo, lo);
	lo = _mm_hadd_ps(lo, lo);
	return (_mm_cvtss_f32(lo));
}
# endif

/*
** Dot product: BF16 vector × F32 vector
** Returns scalar float
** SIMD: processes 8 elements per iteration
*/
static inline float	simd_dot_bf16_f32(const t_bf16 *a, const float *b, int n)
{
	float	sum;
	int		i;

# if SIMD_ENABLED
	__m256	sum_vec;
	__m128i	bf16_a;
	__m256i	a_32;
	__m256	a_f32;
	__m256	b_vec;

	sum_vec = _mm256_setzero_ps();
	i = 0;
	while (i + 7 < n)
	{
		bf16_a = _mm_loadu_si128((__m128i *)(a + i));
		a_32 = _mm256_cvtepu16_epi32(bf16_a);
		a_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(a_32, 16));
		b_vec = _mm256_loadu_ps(b + i);
		sum_vec = _mm256_fmadd_ps(a_f32, b_vec, sum_vec);
		i += 8;
	}
	sum = simd_hsum_ps(sum_vec);
# else
	sum = 0.0f;
	i = 0;
# endif
	while (i < n)
	{
		sum += bf16_to_float(a[i]) * b[i];
		i++;
	}
	return (sum);
}

/*
** Dot product: F32 vector × F32 vector
** Returns scalar float
*/
static inline float	simd_dot_f32_f32(const float *a, const float *b, int n)
{
	float	sum;
	int		i;

# if SIMD_ENABLED
	__m256	sum_vec;

	sum_vec = _mm256_setzero_ps();
	i = 0;
	while (i + 7 < n)
	{
		sum_vec = _mm256_fmadd_ps(
			_mm256_loadu_ps(a + i),
			_mm256_loadu_ps(b + i),
			sum_vec);
		i += 8;
	}
	sum = simd_hsum_ps(sum_vec);
# else
	sum = 0.0f;
	i = 0;
# endif
	while (i < n)
	{
		sum += a[i] * b[i];
		i++;
	}
	return (sum);
}

/*
** Fused multiply-add: out[i] += scale * bf16_vec[i]
** Converts BF16 to F32, scales, accumulates into F32 output
*/
static inline void	simd_fma_bf16_to_f32(float *out, const t_bf16 *vec,
						float scale, int n)
{
	int	i;

# if SIMD_ENABLED
	__m256	scale_vec;
	__m128i	bf16_v;
	__m256i	v_32;
	__m256	v_f32;
	__m256	out_vec;

	scale_vec = _mm256_set1_ps(scale);
	i = 0;
	while (i + 7 < n)
	{
		bf16_v = _mm_loadu_si128((__m128i *)(vec + i));
		v_32 = _mm256_cvtepu16_epi32(bf16_v);
		v_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(v_32, 16));
		out_vec = _mm256_loadu_ps(out + i);
		out_vec = _mm256_fmadd_ps(scale_vec, v_f32, out_vec);
		_mm256_storeu_ps(out + i, out_vec);
		i += 8;
	}
# else
	i = 0;
# endif
	while (i < n)
	{
		out[i] += scale * bf16_to_float(vec[i]);
		i++;
	}
}

/*
** Scale F32 vector in place: vec[i] *= scale
*/
static inline void	simd_scale_f32(float *vec, float scale, int n)
{
	int	i;

# if SIMD_ENABLED
	__m256	scale_vec;

	scale_vec = _mm256_set1_ps(scale);
	i = 0;
	while (i + 7 < n)
	{
		_mm256_storeu_ps(vec + i,
			_mm256_mul_ps(_mm256_loadu_ps(vec + i), scale_vec));
		i += 8;
	}
# else
	i = 0;
# endif
	while (i < n)
	{
		vec[i] *= scale;
		i++;
	}
}

/*
** Pack 8x 32-bit ints to 8x 16-bit ints (lower 16 bits of each)
** Used for FP32 -> BF16 conversion with rounding
*/
# if SIMD_ENABLED
static inline __m128i	simd_cvtepi32_epi16(__m256i a)
{
	__m128i	lo;
	__m128i	hi;

	lo = _mm256_castsi256_si128(a);
	hi = _mm256_extracti128_si256(a, 1);
	return (_mm_packus_epi32(lo, hi));
}
# endif

/*
** Convert F32 to BF16 with proper rounding (not truncation)
** Stores n BF16 values from n F32 values
*/
static inline void	simd_f32_to_bf16(t_bf16 *out, const float *in, int n)
{
	int	i;

# if SIMD_ENABLED
	__m256i		f32_bits;
	__m256i		rounding;
	__m256i		bf16_bits;
	__m128i		result;

	rounding = _mm256_set1_epi32(0x8000);
	i = 0;
	while (i + 7 < n)
	{
		f32_bits = _mm256_castps_si256(_mm256_loadu_ps(in + i));
		f32_bits = _mm256_add_epi32(f32_bits, rounding);
		bf16_bits = _mm256_srli_epi32(f32_bits, 16);
		result = simd_cvtepi32_epi16(bf16_bits);
		_mm_storeu_si128((__m128i *)(out + i), result);
		i += 8;
	}
# else
	i = 0;
# endif
	while (i < n)
	{
		out[i] = float_to_bf16(in[i]);
		i++;
	}
}

#endif
