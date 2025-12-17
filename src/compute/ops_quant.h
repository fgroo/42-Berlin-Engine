/* ************************************************************************** */
/*                                                                            */
/*   ops_quant.h - FP8/INT8 Quantization for DeepSeek-style Memory Efficiency */
/*                                                                            */
/*   Phase 2: Project Deep Freeze                                             */
/*   - FP8 E4M3 for Lightning Indexer (halves bandwidth)                      */
/*   - INT8 Block Quantization for KV Cache                                   */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_QUANT_H
# define OPS_QUANT_H

# include <stdint.h>

/*
** FP8 E4M3 Format (NVIDIA/OCP Standard)
** - 1 sign bit, 4 exponent bits, 3 mantissa bits
** - Range: [-448, 448], Precision: ~0.1%
** - Perfect for attention scores and indexer weights
*/
typedef uint8_t		t_fp8_e4m3;

/*
** INT8 Symmetric Quantization
** - Range: [-127, 127] with scale factor
** - Used for KV cache block quantization
*/
typedef int8_t		t_qint8;

/*
** Quantized Indexer Parameters
** Used for DeepSeek Lightning Indexer in FP8 format
*/
typedef struct s_quant_indexer
{
	t_fp8_e4m3	*q_fp8;      /* [head_dim] - Quantized query */
	t_fp8_e4m3	*k_fp8;      /* [head_dim] - Quantized key */
	float		scale;        /* Dequantization factor */
	int			head_dim;
}	t_quant_indexer;

/*
** Quantized KV Block (INT8 with per-token scaling)
** Replaces BF16 blocks for 2x memory savings
*/
typedef struct s_quant_block
{
	t_qint8		*k_data;     /* [block_size * head_dim] */
	t_qint8		*v_data;     /* [block_size * head_dim] */
	float		*k_scales;   /* [block_size] - Per-token scale */
	float		*v_scales;   /* [block_size] - Per-token scale */
	int			head_dim;
	int			n_tokens;
}	t_quant_block;

/*
** FP8 E4M3 Lookup Table (256 entries = 1KB, fits in L1)
** Precomputed at init for zero-cost FP8 -> FP32 conversion
*/
extern float	g_fp8_lut[256];

/*
** Initialize FP8 lookup table - call once at model init
*/
void			quant_init_fp8_lut(void);

/*
** Convert FP32 vector to FP8 E4M3 (for storing)
** @param out: Output FP8 buffer [n]
** @param in: Input FP32 buffer [n]
** @param n: Vector length
** @param scale: Output scaling factor
*/
void			quant_f32_to_fp8(t_fp8_e4m3 *out, const float *in,
					int n, float *scale);

/*
** Compute Lightning Indexer score in FP8 (DeepSeek formula)
** score = sum_d ReLU(q[d] * k[d]) * scale
** @returns: Indexer score for block selection
*/
float			compute_lightning_score_fp8(const t_fp8_e4m3 *q_fp8,
					const t_fp8_e4m3 *k_fp8, float scale, int head_dim);

/*
** AVX2-optimized FP8 score computation (main kernel)
*/
float			compute_lightning_score_fp8_avx2(const t_fp8_e4m3 *q_fp8,
					const t_fp8_e4m3 *k_fp8, float scale, int head_dim);

/*
** INT8 Quantization for KV Cache
** @param out: Output INT8 buffer [n]
** @param in: Input FP32 buffer [n]
** @param n: Vector length
** @returns: Scale factor for dequantization
*/
float			quant_f32_to_int8(t_qint8 *out, const float *in, int n);

/*
** INT8 Dequantization (inline for hot path)
*/
static inline float	dequant_int8(t_qint8 val, float scale)
{
	return ((float)val * scale);
}

/*
** AVX2 INT8 -> FP32 dequantization kernel
** @param out: Output FP32 buffer [n]
** @param in: Input INT8 buffer [n]
** @param scale: Dequantization scale
** @param n: Vector length
*/
void			dequant_int8_to_f32_avx2(float *out, const t_qint8 *in,
					float scale, int n);

#endif
