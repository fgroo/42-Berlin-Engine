/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   simd_kernels.h                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 18:30:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/14 18:30:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef SIMD_KERNELS_H
# define SIMD_KERNELS_H

# include <stdint.h>
# include <string.h>  /* memcpy for simd_bf16_to_f32 scalar fallback */
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
** AVX-512 Detection (512-bit vectors, 32 registers)
** 2x theoretical throughput over AVX2 on supported CPUs (Skylake-X, Zen4)
*/
# ifdef __AVX512F__
#  define SIMD_512_ENABLED 1
# else
#  define SIMD_512_ENABLED 0
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
** Horizontal sum of 16 floats in __m512 register (AVX-512)
** Reduces 512-bit to scalar in log2(16) = 4 steps
*/
# if SIMD_512_ENABLED
static inline float	simd_hsum_512(__m512 v)
{
	__m256	lo256;
	__m256	hi256;

	/* Split 512 -> 2x256 and add */
	lo256 = _mm512_castps512_ps256(v);
	hi256 = _mm512_extractf32x8_ps(v, 1);
	lo256 = _mm256_add_ps(lo256, hi256);
	/* Use existing 256-bit reduction */
	return (simd_hsum_ps(lo256));
}
# endif

/*
** Dot product: BF16 vector × F32 vector
** Returns scalar float
** AVX-512: processes 16 elements per iteration (2x throughput)
** AVX2: processes 8 elements per iteration
*/
static inline float	simd_dot_bf16_f32(const t_bf16 *a, const float *b, int n)
{
	float	sum;
	int		i;

# if SIMD_512_ENABLED
	/* AVX-512 fast path: 16 elements per iteration */
	__m512	sum_vec_512 = _mm512_setzero_ps();
	i = 0;
	while (i + 15 < n)
	{
		/* Load 16x BF16 (256 bits) */
		__m256i bf16_256 = _mm256_loadu_si256((__m256i *)(a + i));
		/* Expand to 16x 32-bit and shift to F32 position */
		__m512i a_512_i = _mm512_cvtepu16_epi32(bf16_256);
		__m512 a_512_f = _mm512_castsi512_ps(_mm512_slli_epi32(a_512_i, 16));
		/* Load 16x F32 */
		__m512 b_512 = _mm512_loadu_ps(b + i);
		/* FMA */
		sum_vec_512 = _mm512_fmadd_ps(a_512_f, b_512, sum_vec_512);
		i += 16;
	}
	sum = simd_hsum_512(sum_vec_512);
	/* Fall through to AVX2 for 8-15 element remainder */
# elif SIMD_ENABLED
	sum = 0.0f;
	i = 0;
# else
	sum = 0.0f;
	i = 0;
# endif

# if SIMD_ENABLED
	/* AVX2 cleanup: processes remaining elements 8 at a time */
	__m256	sum_vec = _mm256_setzero_ps();
	while (i + 7 < n)
	{
		__m128i bf16_a = _mm_loadu_si128((__m128i *)(a + i));
		__m256i a_32 = _mm256_cvtepu16_epi32(bf16_a);
		__m256 a_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(a_32, 16));
		__m256 b_vec = _mm256_loadu_ps(b + i);
		sum_vec = _mm256_fmadd_ps(a_f32, b_vec, sum_vec);
		i += 8;
	}
	sum += simd_hsum_ps(sum_vec);
# endif
	/* Scalar remainder */
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
** FUSED RESCALE + FMA: out = out * scale1 + bf16_vec * scale2
** Used in online softmax attention to avoid separate scale + FMA loops.
** This halves memory bandwidth by reading/writing 'out' only once.
*/
static inline void	simd_rescale_fma_bf16(float *out, float scale1,
				const t_bf16 *vec, float scale2, int n)
{
	int	i;

# if SIMD_ENABLED
	__m256	scale1_vec;
	__m256	scale2_vec;
	__m128i	bf16_v;
	__m256i	v_32;
	__m256	v_f32;
	__m256	out_vec;

	scale1_vec = _mm256_set1_ps(scale1);
	scale2_vec = _mm256_set1_ps(scale2);
	i = 0;
	while (i + 7 < n)
	{
		/* Load and convert BF16 to F32 */
		bf16_v = _mm_loadu_si128((__m128i *)(vec + i));
		v_32 = _mm256_cvtepu16_epi32(bf16_v);
		v_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(v_32, 16));
		/* out = out * scale1 + v_f32 * scale2 */
		out_vec = _mm256_loadu_ps(out + i);
		out_vec = _mm256_mul_ps(out_vec, scale1_vec);
		out_vec = _mm256_fmadd_ps(scale2_vec, v_f32, out_vec);
		_mm256_storeu_ps(out + i, out_vec);
		i += 8;
	}
# else
	i = 0;
# endif
	while (i < n)
	{
		out[i] = out[i] * scale1 + bf16_to_float(vec[i]) * scale2;
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

/*
** SIMD Vector Addition: dst[i] += src[i]
** Used for residual connections (x += xb)
** AVX-512: 16 floats per iteration, AVX2: 8 floats per iteration
*/
static inline void	simd_add_f32(float *dst, const float *src, int n)
{
	int	i;

# if SIMD_512_ENABLED
	/* AVX-512 fast path: 16 floats per iteration */
	i = 0;
	while (i + 15 < n)
	{
		_mm512_storeu_ps(dst + i,
			_mm512_add_ps(_mm512_loadu_ps(dst + i), _mm512_loadu_ps(src + i)));
		i += 16;
	}
	/* Fall through to AVX2 for remainder */
# elif SIMD_ENABLED
	i = 0;
# else
	i = 0;
# endif

# if SIMD_ENABLED
	/* AVX2 cleanup: 8 floats at a time */
	while (i + 7 < n)
	{
		_mm256_storeu_ps(dst + i,
			_mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
		i += 8;
	}
# endif
	/* Scalar remainder */
	while (i < n)
	{
		dst[i] += src[i];
		i++;
	}
}

/*
** SIMD BF16→F32 Conversion (Batch)
** Used for embedding lookup - replaces scalar memcpy loop
** Processes 8 elements per iteration with AVX2
*/
static inline void	simd_bf16_to_f32(float *dst, const uint16_t *src, int n)
{
	int	i;

# if SIMD_ENABLED
	__m128i	bf16_vals;
	__m256i	int32_vals;
	__m256i	f32_bits;

	i = 0;
	while (i + 7 < n)
	{
		/* Load 8x BF16 (128 bits) */
		bf16_vals = _mm_loadu_si128((__m128i *)(src + i));
		/* Expand to 8x 32-bit integers */
		int32_vals = _mm256_cvtepu16_epi32(bf16_vals);
		/* Shift left 16 to get F32 bit pattern */
		f32_bits = _mm256_slli_epi32(int32_vals, 16);
		/* Store as float */
		_mm256_storeu_ps(dst + i, _mm256_castsi256_ps(f32_bits));
		i += 8;
	}
# else
	i = 0;
# endif
	/* Scalar remainder */
	while (i < n)
	{
		uint32_t val = (uint32_t)src[i] << 16;
		memcpy(&dst[i], &val, sizeof(float));
		i++;
	}
}

#endif
