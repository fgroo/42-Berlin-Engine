/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_simd.h                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 14:50:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 14:50:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_SIMD_H
# define OPS_SIMD_H

# include <stddef.h>

/*
** SIMD-optimized compute kernels (AVX2 + FMA)
*/

/* Sum of squares: sum(x[i]^2) */
float	ops_simd_sum_sq(const float *x, size_t n);

/* L2 Norm: sqrt(sum(x[i]^2)) */
float	ops_simd_norm(const float *x, size_t n);

/* Scalar fallbacks for comparison */
float	ops_scalar_sum_sq(const float *x, size_t n);
float	ops_scalar_norm(const float *x, size_t n);

/* Scale vector: out[i] = x[i] * scale */
void	ops_simd_scale(float *out, const float *x, float scale, size_t n);

/*
** ===========================================================================
** BF16 <-> F32 SIMD CONVERSION (Phase 4: GEMM Kernel)
** ===========================================================================
** BF16 (bfloat16) is F32 with the lower 16 mantissa bits truncated.
** Layout: [sign:1][exp:8][mantissa:7] vs F32 [sign:1][exp:8][mantissa:23]
**
** Conversion is just a 16-bit left shift (BF16->F32) or right shift (F32->BF16).
** ===========================================================================
*/

# include <immintrin.h>

/*
** Convert 8 BF16 values (packed in 128-bit) to 8 F32 values (256-bit)
** Method: Zero-extend u16 to u32, then shift left 16 bits.
*/
static inline __m256	mm256_cvtbf16_ps(__m128i v_bf16)
{
	__m256i	v_u32;

	/* Zero-extend 8x u16 to 8x u32 */
	v_u32 = _mm256_cvtepu16_epi32(v_bf16);
	/* Shift left 16 bits - BF16 occupies upper bits of F32 */
	v_u32 = _mm256_slli_epi32(v_u32, 16);
	/* Reinterpret bits as float */
	return (_mm256_castsi256_ps(v_u32));
}

/*
** Convert single BF16 (u16) to F32 - safe scalar version using union
** Avoids strict-aliasing violations.
*/
static inline float	bf16_to_f32_safe(uint16_t bf16)
{
	union { uint32_t u; float f; }	conv;

	conv.u = (uint32_t)bf16 << 16;
	return (conv.f);
}

/*
** Broadcast single BF16 value to all 8 lanes of __m256
*/
static inline __m256	mm256_broadcast_bf16(uint16_t bf16)
{
	return (_mm256_set1_ps(bf16_to_f32_safe(bf16)));
}

#endif
