/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_activation.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 23:15:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/15 23:15:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** FUSED ACTIVATION KERNELS (Phase 6: Operation "Fused Synapse")
** ===========================================================================
** SwiGLU activation: out = SiLU(gate) * up = (gate * sigmoid(gate)) * up
**
** Traditional approach (3 memory roundtrips):
**   1. Compute gate = X @ W_gate -> store
**   2. Compute up = X @ W_up -> store
**   3. Load gate, compute SiLU(gate), store -> load, multiply up, store
**
** Fused approach (1 memory roundtrip):
**   Load gate and up, compute SiLU * up in registers, store result
**
** This eliminates 2 cache roundtrips per element = ~2x speedup on FFN
** ===========================================================================
*/

#include <immintrin.h>
#include <stddef.h>
#include <omp.h>
#include "ops_math_fast.h"

/*
** Fused SiLU-Mul AVX2 Kernel
** out[i] = (gate[i] * sigmoid(gate[i])) * up[i]
**
** Uses fast_expf_avx2 for ~10x faster exp than stdlib
** Processes 8 floats per iteration (256-bit registers)
**
** @param dst:  Output buffer (F32)
** @param gate: Gate projection output (F32)
** @param up:   Up projection output (F32)
** @param n:    Number of elements
*/
void	op_silu_mul_fused_f32(float *dst, const float *gate,
							const float *up, size_t n)
{
	__m256	ones;
	__m256	zero;
	__m256	g;
	__m256	u;
	__m256	neg_g;
	__m256	exp_neg_g;
	__m256	sigmoid;
	__m256	silu;
	__m256	result;
	size_t	i;

	ones = _mm256_set1_ps(1.0f);
	zero = _mm256_setzero_ps();
	i = 0;

	/* Main SIMD loop: 8 floats per iteration */
	while (i + 8 <= n)
	{
		/* Load gate and up vectors */
		g = _mm256_loadu_ps(gate + i);
		u = _mm256_loadu_ps(up + i);

		/* Sigmoid(g) = 1 / (1 + exp(-g)) */
		neg_g = _mm256_sub_ps(zero, g);
		exp_neg_g = fast_expf_avx2(neg_g);
		sigmoid = _mm256_div_ps(ones, _mm256_add_ps(ones, exp_neg_g));

		/* SiLU(g) = g * sigmoid(g) */
		silu = _mm256_mul_ps(g, sigmoid);

		/* Fused multiply: result = silu * up */
		result = _mm256_mul_ps(silu, u);

		/* Store */
		_mm256_storeu_ps(dst + i, result);
		i += 8;
	}

	/* Scalar cleanup for remainder */
	while (i < n)
	{
		float	gv;
		float	uv;
		float	sig;

		gv = gate[i];
		uv = up[i];
		sig = 1.0f / (1.0f + fast_expf(-gv));
		dst[i] = gv * sig * uv;
		i++;
	}
}

/*
** OpenMP-parallelized version for very large dimensions
** Chunks the work across threads, each thread uses SIMD internally
*/
void	op_silu_mul_fused_f32_omp(float *dst, const float *gate,
								const float *up, size_t n)
{
	/* For small n, avoid OMP overhead */
	if (n < 2048)
	{
		op_silu_mul_fused_f32(dst, gate, up, n);
		return ;
	}

	#pragma omp parallel
	{
		int		tid;
		int		nthreads;
		size_t	chunk_size;
		size_t	start;
		size_t	end;

		tid = omp_get_thread_num();
		nthreads = omp_get_num_threads();
		chunk_size = (n + nthreads - 1) / nthreads;
		start = tid * chunk_size;
		end = start + chunk_size;
		if (end > n)
			end = n;
		if (start < n)
			op_silu_mul_fused_f32(dst + start, gate + start, up + start,
				end - start);
	}
}

/*
** BF16 version: Load BF16, compute in F32, store BF16
** Used when weights are in BF16 format (Ministral)
*/
void	op_silu_mul_fused_bf16(uint16_t *dst, const uint16_t *gate,
							const uint16_t *up, size_t n)
{
	__m256	ones;
	__m256	zero;
	__m256	g;
	__m256	u;
	__m256	neg_g;
	__m256	exp_neg_g;
	__m256	sigmoid;
	__m256	silu;
	__m256	result;
	__m128i	gate_bf16;
	__m128i	up_bf16;
	__m128i	result_bf16;
	__m256i	g_u32;
	__m256i	u_u32;
	__m256i	r_u32;
	__m128i	r_lo;
	__m128i	r_hi;
	size_t	i;

	ones = _mm256_set1_ps(1.0f);
	zero = _mm256_setzero_ps();
	i = 0;

	/* Main SIMD loop: 8 BF16 values per iteration */
	while (i + 8 <= n)
	{
		/* Load 8 BF16 values and convert to F32 */
		gate_bf16 = _mm_loadu_si128((const __m128i *)(gate + i));
		up_bf16 = _mm_loadu_si128((const __m128i *)(up + i));

		/* BF16 -> F32: zero-extend to 32-bit, shift left 16 */
		g_u32 = _mm256_cvtepu16_epi32(gate_bf16);
		g_u32 = _mm256_slli_epi32(g_u32, 16);
		g = _mm256_castsi256_ps(g_u32);

		u_u32 = _mm256_cvtepu16_epi32(up_bf16);
		u_u32 = _mm256_slli_epi32(u_u32, 16);
		u = _mm256_castsi256_ps(u_u32);

		/* Sigmoid(g) = 1 / (1 + exp(-g)) */
		neg_g = _mm256_sub_ps(zero, g);
		exp_neg_g = fast_expf_avx2(neg_g);
		sigmoid = _mm256_div_ps(ones, _mm256_add_ps(ones, exp_neg_g));

		/* SiLU(g) = g * sigmoid(g) */
		silu = _mm256_mul_ps(g, sigmoid);

		/* Fused multiply: result = silu * up */
		result = _mm256_mul_ps(silu, u);

		/* F32 -> BF16: shift right 16, pack */
		r_u32 = _mm256_castps_si256(result);
		r_u32 = _mm256_srli_epi32(r_u32, 16);
		r_lo = _mm256_castsi256_si128(r_u32);
		r_hi = _mm256_extracti128_si256(r_u32, 1);
		result_bf16 = _mm_packus_epi32(r_lo, r_hi);

		/* Store 8 BF16 values */
		_mm_storeu_si128((__m128i *)(dst + i), result_bf16);
		i += 8;
	}

	/* Scalar cleanup */
	while (i < n)
	{
		union { uint32_t u; float f; }	conv_g;
		union { uint32_t u; float f; }	conv_u;
		union { uint32_t u; float f; }	conv_r;
		float	gv;
		float	uv;
		float	sig;
		float	res;

		conv_g.u = (uint32_t)gate[i] << 16;
		conv_u.u = (uint32_t)up[i] << 16;
		gv = conv_g.f;
		uv = conv_u.f;
		sig = 1.0f / (1.0f + fast_expf(-gv));
		res = gv * sig * uv;
		conv_r.f = res;
		dst[i] = (uint16_t)(conv_r.u >> 16);
		i++;
	}
}

/*
** Suppress unused function warnings
*/
void	_ops_activation_suppress_unused(void)
{
	(void)op_silu_mul_fused_f32_omp;
	(void)op_silu_mul_fused_bf16;
}
