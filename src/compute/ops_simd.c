/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_simd.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 14:50:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/15 14:50:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** SIMD COMPUTE KERNELS (Phase 3: Raw Performance)
** ===========================================================================
** AVX2 + FMA optimized kernels for gradient operations.
** These replace scalar loops in the nested learning backward pass.
**
** Key optimizations:
** - Process 8 floats per cycle (256-bit registers)
** - FMA (Fused Multiply-Add) reduces latency
** - Unaligned loads for safety (minimal penalty on Haswell+)
** - Multiple accumulators hide FMA latency (4 cycles on Skylake)
** ===========================================================================
*/

#include <immintrin.h>
#include <math.h>
#include <stddef.h>

/*
** Sum of squares: sum(x[i]^2) for i in [0, n)
** This is the first step of L2 norm computation.
**
** Uses 4 accumulators to hide FMA latency (4 cycles on modern Intel).
** Processes 32 floats (128 bytes) per main loop iteration.
**
** @param x: Input vector (F32)
** @param n: Number of elements
** @return: Sum of squared elements
*/
float	ops_simd_sum_sq(const float *x, size_t n)
{
	__m256	vsum0;
	__m256	vsum1;
	__m256	vsum2;
	__m256	vsum3;
	__m256	vx0;
	__m256	vx1;
	__m256	vx2;
	__m256	vx3;
	__m256	vtmp;
	__m128	lo;
	float	sum;
	size_t	i;

	/* Initialize 4 accumulators to zero */
	vsum0 = _mm256_setzero_ps();
	vsum1 = _mm256_setzero_ps();
	vsum2 = _mm256_setzero_ps();
	vsum3 = _mm256_setzero_ps();
	i = 0;

	/* Main loop: 32 floats per iteration (4 x 8) */
	while (i + 32 <= n)
	{
		/* Unaligned loads (safe for any pointer alignment) */
		vx0 = _mm256_loadu_ps(x + i);
		vx1 = _mm256_loadu_ps(x + i + 8);
		vx2 = _mm256_loadu_ps(x + i + 16);
		vx3 = _mm256_loadu_ps(x + i + 24);
		/* FMA: vsum += x * x */
		vsum0 = _mm256_fmadd_ps(vx0, vx0, vsum0);
		vsum1 = _mm256_fmadd_ps(vx1, vx1, vsum1);
		vsum2 = _mm256_fmadd_ps(vx2, vx2, vsum2);
		vsum3 = _mm256_fmadd_ps(vx3, vx3, vsum3);
		i += 32;
	}

	/* Cleanup: 8 floats at a time */
	while (i + 8 <= n)
	{
		vx0 = _mm256_loadu_ps(x + i);
		vsum0 = _mm256_fmadd_ps(vx0, vx0, vsum0);
		i += 8;
	}

	/* Combine 4 accumulators */
	vsum0 = _mm256_add_ps(vsum0, vsum1);
	vsum2 = _mm256_add_ps(vsum2, vsum3);
	vsum0 = _mm256_add_ps(vsum0, vsum2);

	/* Horizontal reduction: 8 floats -> 1 float */
	/* Step 1: Swap hi/lo 128-bit lanes and add */
	vtmp = _mm256_permute2f128_ps(vsum0, vsum0, 1);
	vsum0 = _mm256_add_ps(vsum0, vtmp);
	/* Step 2: Horizontal adds within 128-bit lane */
	vsum0 = _mm256_hadd_ps(vsum0, vsum0);
	vsum0 = _mm256_hadd_ps(vsum0, vsum0);
	/* Extract scalar result */
	lo = _mm256_castps256_ps128(vsum0);
	sum = _mm_cvtss_f32(lo);

	/* Scalar remainder (< 8 elements) */
	while (i < n)
	{
		sum += x[i] * x[i];
		i++;
	}
	return (sum);
}

/*
** L2 Norm: sqrt(sum(x[i]^2))
** Wrapper around sum_sq with final sqrt.
*/
float	ops_simd_norm(const float *x, size_t n)
{
	float	sum;

	sum = ops_simd_sum_sq(x, n);
	return (sqrtf(sum));
}

/*
** Scalar fallback for comparison/testing
*/
float	ops_scalar_sum_sq(const float *x, size_t n)
{
	float	sum;
	size_t	i;

	sum = 0.0f;
	i = 0;
	while (i < n)
	{
		sum += x[i] * x[i];
		i++;
	}
	return (sum);
}

float	ops_scalar_norm(const float *x, size_t n)
{
	return (sqrtf(ops_scalar_sum_sq(x, n)));
}

/*
** ===========================================================================
** BF16 <-> F32 CONVERSION (SIMD)
** ===========================================================================
** BF16 is F32 with the lower 16 bits truncated.
** This is a fast approximation (truncation, not round-to-nearest-even).
** ===========================================================================
*/

/*
** Convert 8 F32 values to 8 BF16 values (packed into __m128i)
** Uses simple truncation: just take upper 16 bits of each F32.
*/
static inline __m128i	f32_to_bf16_8x(__m256 f32)
{
	__m256i	as_int;
	__m256i	shifted;
	__m128i	lo;
	__m128i	hi;

	/* Reinterpret F32 as int32, shift right 16 bits */
	as_int = _mm256_castps_si256(f32);
	shifted = _mm256_srli_epi32(as_int, 16);
	/* Pack 8x32-bit to 8x16-bit */
	lo = _mm256_castsi256_si128(shifted);
	hi = _mm256_extracti128_si256(shifted, 1);
	return (_mm_packus_epi32(lo, hi));
}

/*
** Scale a vector by a scalar using SIMD
** out[i] = x[i] * scale
*/
void	ops_simd_scale(float *out, const float *x, float scale, size_t n)
{
	__m256	vscale;
	__m256	vx;
	size_t	i;

	vscale = _mm256_set1_ps(scale);
	i = 0;
	while (i + 8 <= n)
	{
		vx = _mm256_loadu_ps(x + i);
		_mm256_storeu_ps(out + i, _mm256_mul_ps(vx, vscale));
		i += 8;
	}
	while (i < n)
	{
		out[i] = x[i] * scale;
		i++;
	}
}

/*
** Suppress unused function warning for f32_to_bf16_8x (used later)
*/
void	_ops_simd_suppress_unused(void)
{
	(void)f32_to_bf16_8x;
}
