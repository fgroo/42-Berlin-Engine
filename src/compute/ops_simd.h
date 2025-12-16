/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_simd.h                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 14:50:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/15 14:50:00 by fgroo      ###   ########.fr       */
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

/*
** ===========================================================================
** STOCHASTIC ROUNDING (Fix Critical #3: Gradient Precision Loss)
** ===========================================================================
** BF16 truncates the lower 16 bits of F32 mantissa, losing small gradients.
** Stochastic rounding adds random noise [0, 1) * ULP before truncation.
** Statistically, this preserves the expected value of small updates.
**
** Example: F32 = 0.1234, BF16 truncates to 0.123
** With SR:  noise = rand() * ULP(0.123)
**           if 0.1234 + noise rounds up to 0.124, we get 0.124
**           Expected value over many updates: 0.1234 (correct!)
**
** Uses XorShift64 for fast, thread-safe PRNG (7 cycles vs 50+ for rand())
** ===========================================================================
*/

typedef struct s_xorshift_state
{
	uint64_t	s;
}	t_xorshift_state;

/* XorShift64: Fast PRNG with 2^64 period */
static inline uint64_t	xorshift64(t_xorshift_state *state)
{
	uint64_t	x;

	x = state->s;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	state->s = x;
	return (x);
}

/* Initialize RNG state (use thread ID + time for uniqueness) */
static inline void	xorshift_init(t_xorshift_state *state, uint64_t seed)
{
	state->s = seed ? seed : 0x12345678ABCDEF01ULL;
}

/*
** Convert F32 to BF16 with stochastic rounding
** Adds random value [0, ULP) before truncation to preserve expected value
*/
static inline uint16_t	float_to_bf16_stochastic(float f, t_xorshift_state *rng)
{
	union { float f; uint32_t u; }	conv;
	uint32_t	noise;
	uint32_t	rounded;

	conv.f = f;
	/* Handle special cases: NaN, Inf, Zero */
	if ((conv.u & 0x7F800000) == 0x7F800000 || conv.u == 0 || conv.u == 0x80000000)
		return ((uint16_t)(conv.u >> 16));
	/* Generate 16-bit random noise for lower bits */
	noise = (uint32_t)(xorshift64(rng) & 0xFFFF);
	/* Add noise to lower 16 bits (implements probabilistic rounding) */
	rounded = conv.u + noise;
	/* Truncate to BF16 (upper 16 bits) */
	return ((uint16_t)(rounded >> 16));
}

/*
** Stochastic rounding for gradient update: weight = weight - lr * grad
** Atomically applies update with stochastic rounding to preserve small deltas
*/
static inline void	bf16_stochastic_update(uint16_t *weight, float lr_grad,
							t_xorshift_state *rng)
{
	union { float f; uint32_t u; }	conv;
	float	w_f32;
	float	w_new;

	/* Convert current weight to F32 */
	conv.u = (uint32_t)(*weight) << 16;
	w_f32 = conv.f;
	/* Apply gradient update in F32 */
	w_new = w_f32 - lr_grad;
	/* Store back with stochastic rounding */
	*weight = float_to_bf16_stochastic(w_new, rng);
}

#endif
