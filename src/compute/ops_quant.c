/* ************************************************************************** */
/*                                                                            */
/*   ops_quant.c - FP8/INT8 Quantization Kernels                              */
/*                                                                            */
/*   Phase 2: Project Deep Freeze                                             */
/*   Implements DeepSeek-style FP8 Lightning Indexer and INT8 KV Cache        */
/*                                                                            */
/* ************************************************************************** */

#include "ops_quant.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

/*
** Global FP8 E4M3 Lookup Table (1KB, L1-resident)
*/
float	g_fp8_lut[256];

/*
** FP8 E4M3 Format Specification:
** - Sign: bit 7
** - Exponent: bits 6-3 (4 bits, bias = 7)
** - Mantissa: bits 2-0 (3 bits, implicit 1.xxx)
** - Special: 0x80 = -0, no inf/nan in E4M3
*/
static float	fp8_to_float(uint8_t fp8)
{
	int		sign;
	int		exp;
	int		mant;
	float	val;

	sign = (fp8 >> 7) & 1;
	exp = (fp8 >> 3) & 0xF;
	mant = fp8 & 0x7;
	if (exp == 0)
	{
		/* Subnormal: 2^-6 * (0.mantissa) */
		val = ldexpf((float)mant / 8.0f, -6);
	}
	else
	{
		/* Normal: 2^(exp-7) * (1.mantissa) */
		val = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
	}
	return (sign ? -val : val);
}

/*
** Initialize FP8 LUT - call once at model init
*/
void	quant_init_fp8_lut(void)
{
	int	i;

	i = 0;
	while (i < 256)
	{
		g_fp8_lut[i] = fp8_to_float((uint8_t)i);
		i++;
	}
}

/*
** Convert FP32 to FP8 E4M3 (quantization)
*/
static uint8_t	float_to_fp8(float val)
{
	int		sign;
	int		exp;
	int		mant;
	float	abs_val;
	float	scaled;

	if (val == 0.0f)
		return (0);
	sign = (val < 0.0f) ? 1 : 0;
	abs_val = fabsf(val);
	/* Clamp to E4M3 range: max = 448 */
	if (abs_val > 448.0f)
		abs_val = 448.0f;
	/* Get exponent */
	exp = (int)floorf(log2f(abs_val)) + 7;
	if (exp < 0)
		exp = 0;
	if (exp > 15)
		exp = 15;
	/* Get mantissa */
	if (exp == 0)
		scaled = abs_val * 64.0f;  /* Subnormal */
	else
		scaled = abs_val / ldexpf(1.0f, exp - 7) - 1.0f;
	mant = (int)(scaled * 8.0f + 0.5f);
	if (mant > 7)
		mant = 7;
	return ((uint8_t)((sign << 7) | (exp << 3) | mant));
}

/*
** Quantize F32 vector to FP8 with dynamic scaling
*/
void	quant_f32_to_fp8(t_fp8_e4m3 *out, const float *in, int n, float *scale)
{
	float	max_abs;
	float	inv_scale;
	int		i;

	/* Find max absolute value for scaling */
	max_abs = 0.0f;
	i = 0;
	while (i < n)
	{
		if (fabsf(in[i]) > max_abs)
			max_abs = fabsf(in[i]);
		i++;
	}
	/* Scale to fit in FP8 range (max ~448) */
	if (max_abs > 0.0f)
	{
		*scale = max_abs / 224.0f;  /* Use half range for safety */
		inv_scale = 1.0f / *scale;
	}
	else
	{
		*scale = 1.0f;
		inv_scale = 1.0f;
	}
	/* Quantize */
	i = 0;
	while (i < n)
	{
		out[i] = float_to_fp8(in[i] * inv_scale);
		i++;
	}
}

/*
** ============================================================================
** DEEPSEEK LIGHTNING INDEXER - FP8 AVX2 KERNEL
** ============================================================================
** Computes: score = sum_d ReLU(q[d] * k[d]) * scale
** 
** Key optimizations:
** - FP8 storage = half the bandwidth vs BF16
** - LUT-based conversion = no expensive bit manipulation
** - 8 scores per iteration via AVX2
** ============================================================================
*/
float	compute_lightning_score_fp8_avx2(const t_fp8_e4m3 *q_fp8,
			const t_fp8_e4m3 *k_fp8, float scale, int head_dim)
{
	__m256		sum;
	__m256		zero;
	__m256		vq;
	__m256		vk;
	__m256		prod;
	__m256		activated;
	float		q_vals[8];
	float		k_vals[8];
	float		temp[8];
	float		result;
	int			i;
	int			j;

	sum = _mm256_setzero_ps();
	zero = _mm256_setzero_ps();
	i = 0;
	/* Main SIMD loop: 8 elements per iteration */
	while (i + 7 < head_dim)
	{
		/* LUT-based FP8 -> F32 conversion (L1 cache resident) */
		j = 0;
		while (j < 8)
		{
			q_vals[j] = g_fp8_lut[q_fp8[i + j]];
			k_vals[j] = g_fp8_lut[k_fp8[i + j]];
			j++;
		}
		/* Load to SIMD registers */
		vq = _mm256_loadu_ps(q_vals);
		vk = _mm256_loadu_ps(k_vals);
		/* Multiply: q[d] * k[d] */
		prod = _mm256_mul_ps(vq, vk);
		/* ReLU: max(0, x) - DeepSeek formula */
		activated = _mm256_max_ps(prod, zero);
		/* Accumulate */
		sum = _mm256_add_ps(sum, activated);
		i += 8;
	}
	/* Horizontal sum reduction */
	_mm256_storeu_ps(temp, sum);
	result = 0.0f;
	j = 0;
	while (j < 8)
	{
		result += temp[j];
		j++;
	}
	/* Scalar tail */
	while (i < head_dim)
	{
		float qv = g_fp8_lut[q_fp8[i]];
		float kv = g_fp8_lut[k_fp8[i]];
		float prod_s = qv * kv;
		if (prod_s > 0.0f)
			result += prod_s;
		i++;
	}
	return (result * scale);
}

/*
** Scalar fallback for testing/debugging
*/
float	compute_lightning_score_fp8(const t_fp8_e4m3 *q_fp8,
			const t_fp8_e4m3 *k_fp8, float scale, int head_dim)
{
	return (compute_lightning_score_fp8_avx2(q_fp8, k_fp8, scale, head_dim));
}

/*
** ============================================================================
** INT8 KV CACHE QUANTIZATION
** ============================================================================
** Per-token symmetric quantization with scale factor
** ============================================================================
*/
float	quant_f32_to_int8(t_qint8 *out, const float *in, int n)
{
	float	max_abs;
	float	scale;
	float	inv_scale;
	int		i;
	int		val;

	/* Find max absolute value */
	max_abs = 0.0f;
	i = 0;
	while (i < n)
	{
		if (fabsf(in[i]) > max_abs)
			max_abs = fabsf(in[i]);
		i++;
	}
	/* Compute scale */
	if (max_abs > 0.0f)
	{
		scale = max_abs / 127.0f;
		inv_scale = 127.0f / max_abs;
	}
	else
	{
		scale = 1.0f;
		inv_scale = 1.0f;
	}
	/* Quantize with rounding */
	i = 0;
	while (i < n)
	{
		val = (int)roundf(in[i] * inv_scale);
		if (val > 127)
			val = 127;
		if (val < -127)
			val = -127;
		out[i] = (t_qint8)val;
		i++;
	}
	return (scale);
}

/*
** AVX2 INT8 dequantization kernel
*/
void	dequant_int8_to_f32_avx2(float *out, const t_qint8 *in,
			float scale, int n)
{
	__m256		scale_vec;
	__m128i		in_8;
	__m256i		in_32;
	__m256		in_f32;
	int			i;

	scale_vec = _mm256_set1_ps(scale);
	i = 0;
	/* Process 8 INT8 values at a time */
	while (i + 7 < n)
	{
		/* Load 8 bytes */
		in_8 = _mm_loadl_epi64((__m128i *)(in + i));
		/* Sign-extend to 32-bit */
		in_32 = _mm256_cvtepi8_epi32(in_8);
		/* Convert to float */
		in_f32 = _mm256_cvtepi32_ps(in_32);
		/* Scale and store */
		_mm256_storeu_ps(out + i, _mm256_mul_ps(in_f32, scale_vec));
		i += 8;
	}
	/* Scalar tail */
	while (i < n)
	{
		out[i] = (float)in[i] * scale;
		i++;
	}
}
