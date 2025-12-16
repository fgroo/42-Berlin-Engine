/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_attention.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 18:30:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/14 18:30:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_attention.h"
#include "simd_kernels.h"
#include "ops_math_fast.h"
#include <string.h>
#include <math.h>
#include <omp.h>

/*
** ============================================================================
** FLASH ATTENTION V1: ONLINE SOFTMAX
** ============================================================================
** Classic "Online Softmax" algorithm from FlashAttention paper.
** Computes attention in a single pass without storing all scores.
**
** Memory: O(head_dim) instead of O(seq_len)
** Numerically stable via running max tracking
**
** Algorithm:
**   1. For each position t:
**      - score_t = Q · K_t * scale
**      - Update running max: max_new = max(max_old, score_t)
**      - Rescale previous sum: sum *= exp(max_old - max_new)
**      - Add current: sum += exp(score_t - max_new)
**      - Rescale output: out *= exp(max_old - max_new)
**      - Accumulate: out += V_t * exp(score_t - max_new)
**   2. Final: out /= sum
** ============================================================================
*/
void	flash_attention_head(float *out, const float *q,
		const t_bf16 *k_cache, const t_bf16 *v_cache,
		int seq_len, int head_dim, float scale)
{
	float	max_score;
	float	sum_exp;
	float	prev_max;
	float	score;
	float	scale_factor;
	float	raw_exp;
	int		t;

	max_score = -INFINITY;
	sum_exp = 0.0f;
	memset(out, 0, head_dim * sizeof(float));
	t = 0;
	while (t < seq_len)
	{
		/* 1. Compute Q · K_t using SIMD */
		score = simd_dot_bf16_f32(k_cache + t * head_dim, q, head_dim);
		score *= scale;
		/* 2. Online softmax update */
		prev_max = max_score;
		if (score > prev_max)
			max_score = score;
		else
			max_score = prev_max;
		scale_factor = fast_expf(prev_max - max_score);
		raw_exp = fast_expf(score - max_score);
		sum_exp = sum_exp * scale_factor + raw_exp;
		/* 3. Fused rescale and accumulate output (half the memory ops) */
		simd_rescale_fma_bf16(out, scale_factor, v_cache + t * head_dim,
			raw_exp, head_dim);
		t++;
	}
	/* 4. Final normalization */
	if (sum_exp > 0.0f)
		simd_scale_f32(out, 1.0f / sum_exp, head_dim);
}

/*
** Dense attention helper: Flash attention over all positions
*/
static void	attention_head_dense(float *out, const float *q,
			const t_bf16 *k_data, const t_bf16 *v_data,
			int kv_len, int kv_stride, int kv_h, int head_dim, float scale)
{
	float	max_score;
	float	sum_exp;
	float	prev_max;
	float	score;
	float	scale_factor;
	float	raw_exp;
	int		t;

	max_score = -INFINITY;
	sum_exp = 0.0f;
	memset(out, 0, head_dim * sizeof(float));
	t = 0;
	while (t < kv_len)
	{
		score = simd_dot_bf16_f32(
			k_data + t * kv_stride + kv_h * head_dim, q, head_dim);
		score *= scale;
		prev_max = max_score;
		if (score > prev_max)
			max_score = score;
		else
			max_score = prev_max;
		scale_factor = fast_expf(prev_max - max_score);
		raw_exp = fast_expf(score - max_score);
		sum_exp = sum_exp * scale_factor + raw_exp;
		/* Fused rescale + FMA (2x less memory traffic) */
		simd_rescale_fma_bf16(out, scale_factor,
			v_data + t * kv_stride + kv_h * head_dim, raw_exp, head_dim);
		t++;
	}
	if (sum_exp > 0.0f)
		simd_scale_f32(out, 1.0f / sum_exp, head_dim);
}

/*
** Sparse attention helper: Flash attention over selected positions only
*/
static void	attention_head_sparse(float *out, const float *q,
			const t_bf16 *k_data, const t_bf16 *v_data,
			const int *indices, int attend_k,
			int kv_stride, int kv_h, int head_dim, float scale)
{
	float	max_score;
	float	sum_exp;
	float	prev_max;
	float	score;
	float	scale_factor;
	float	raw_exp;
	int		i;
	int		seq_idx;

	max_score = -INFINITY;
	sum_exp = 0.0f;
	memset(out, 0, head_dim * sizeof(float));
	i = 0;
	while (i < attend_k)
	{
		seq_idx = indices[i];
		score = simd_dot_bf16_f32(
			k_data + seq_idx * kv_stride + kv_h * head_dim, q, head_dim);
		score *= scale;
		prev_max = max_score;
		if (score > prev_max)
			max_score = score;
		else
			max_score = prev_max;
		scale_factor = fast_expf(prev_max - max_score);
		raw_exp = fast_expf(score - max_score);
		sum_exp = sum_exp * scale_factor + raw_exp;
		/* Fused rescale + FMA */
		simd_rescale_fma_bf16(out, scale_factor,
			v_data + seq_idx * kv_stride + kv_h * head_dim, raw_exp, head_dim);
		i++;
	}
	if (sum_exp > 0.0f)
		simd_scale_f32(out, 1.0f / sum_exp, head_dim);
}

/*
** Multi-head dense attention with GQA support
*/
void	op_attention_dense(t_attention_params *p, int kv_len)
{
	t_bf16	*k_data;
	t_bf16	*v_data;
	int		kv_stride;
	int		h;
	int		kv_h;

	k_data = (t_bf16 *)p->kv_cache->k.data;
	v_data = (t_bf16 *)p->kv_cache->v.data;
	kv_stride = p->n_kv_heads * p->head_dim;
	memset(p->output, 0, p->n_heads * p->head_dim * sizeof(float));
	#pragma omp parallel for schedule(static) private(kv_h)
	for (h = 0; h < p->n_heads; h++)
	{
		kv_h = h / (p->n_heads / p->n_kv_heads);
		attention_head_dense(
			p->output + h * p->head_dim,
			p->q + h * p->head_dim,
			k_data, v_data,
			kv_len, kv_stride, kv_h, p->head_dim, p->scale);
	}
}

/*
** Multi-head sparse attention with GQA support
*/
void	op_attention_sparse(t_attention_params *p)
{
	t_bf16	*k_data;
	t_bf16	*v_data;
	int		kv_stride;
	int		h;
	int		kv_h;

	k_data = (t_bf16 *)p->kv_cache->k.data;
	v_data = (t_bf16 *)p->kv_cache->v.data;
	kv_stride = p->n_kv_heads * p->head_dim;
	memset(p->output, 0, p->n_heads * p->head_dim * sizeof(float));
	#pragma omp parallel for schedule(static) private(kv_h)
	for (h = 0; h < p->n_heads; h++)
	{
		kv_h = h / (p->n_heads / p->n_kv_heads);
		attention_head_sparse(
			p->output + h * p->head_dim,
			p->q + h * p->head_dim,
			k_data, v_data,
			p->topk_idx, p->attend_k,
			kv_stride, kv_h, p->head_dim, p->scale);
	}
}

/*
** High-level attention dispatcher
*/
void	op_multihead_attention(t_attention_params *p)
{
	int	kv_len;

	kv_len = p->kv_cache->current_seq_len;
	if (p->topk_idx != NULL && p->attend_k > 0 && p->attend_k < kv_len)
		op_attention_sparse(p);
	else
		op_attention_dense(p, kv_len);
}

/*
** ============================================================================
** BLOCK-BASED FLASH ATTENTION (Paged KV Cache)
** ============================================================================
** For paged cache, we iterate over blocks instead of linear positions.
** Block layout: [n_kv_heads][BLOCK_SIZE][head_dim]
** This enables:
** - Contiguous SIMD access within block
** - O(K) sparse attention by selecting only relevant blocks
** - Cache-friendly prefetching
** ============================================================================
*/

#include "memory/paged.h"

/*
** Flash attention over a single block for one head
** @param out: Output accumulator [head_dim] (modified in place)
** @param q: Query vector [head_dim]
** @param block: Current KV block
** @param kv_h: Which KV head to use
** @param head_dim: Dimension per head
** @param scale: 1/sqrt(head_dim)
** @param max_score: Running max (in/out)
** @param sum_exp: Running sum of exp (in/out)
*/
static void	attention_block(float *out, const float *q,
			t_kv_block *block, int kv_h, int head_dim, float scale,
			float *max_score, float *sum_exp)
{
	int		n_tokens;
	int		t;
	int		offset;
	float	score;
	float	prev_max;
	float	scale_factor;
	float	raw_exp;
	t_bf16	*k_ptr;
	t_bf16	*v_ptr;

	n_tokens = block->n_tokens;
	if (n_tokens <= 0)
		return ;
	/* Block layout: [kv_head][token][dim] */
	/* Offset to this head's data: kv_h * BLOCK_SIZE * head_dim */
	offset = kv_h * KV_BLOCK_SIZE * head_dim;
	t = 0;
	while (t < n_tokens)
	{
		k_ptr = block->k + offset + t * head_dim;
		v_ptr = block->v + offset + t * head_dim;
		/* Compute Q · K_t */
		score = simd_dot_bf16_f32(k_ptr, q, head_dim) * scale;
		/* Online softmax update */
		prev_max = *max_score;
		if (score > prev_max)
			*max_score = score;
		scale_factor = fast_expf(prev_max - *max_score);
		raw_exp = fast_expf(score - *max_score);
		*sum_exp = (*sum_exp) * scale_factor + raw_exp;
		/* Fused rescale + FMA */
		simd_rescale_fma_bf16(out, scale_factor, v_ptr, raw_exp, head_dim);
		t++;
	}
}

/*
** Paged attention for a single head - iterates over all blocks
*/
static void	attention_head_paged(float *out, const float *q,
			t_paged_kv_cache *pkv, int kv_h, int head_dim, float scale)
{
	t_block_manager	*bm;
	int				page_base;
	int				n_blocks;
	int				block_idx;
	int				phys_idx;
	float			max_score;
	float			sum_exp;

	bm = pkv->bm;
	page_base = pkv->layer_idx * bm->max_logical;
	n_blocks = bm->logical_len[pkv->layer_idx];
	max_score = -INFINITY;
	sum_exp = 0.0f;
	memset(out, 0, head_dim * sizeof(float));
	block_idx = 0;
	while (block_idx < n_blocks)
	{
		phys_idx = bm->page_table[page_base + block_idx];
		if (phys_idx >= 0)
		{
			attention_block(out, q, &bm->blocks[phys_idx],
				kv_h, head_dim, scale, &max_score, &sum_exp);
		}
		block_idx++;
	}
	/* Final normalization */
	if (sum_exp > 0.0f)
		simd_scale_f32(out, 1.0f / sum_exp, head_dim);
}

/*
** Paged sparse attention - iterates over selected blocks only
** @param selected_blocks: Array of physical block indices to attend to
** @param n_selected: Number of selected blocks
*/
static void	attention_head_paged_sparse(float *out, const float *q,
			t_block_manager *bm, const int *selected_blocks, int n_selected,
			int kv_h, int head_dim, float scale)
{
	int		i;
	int		phys_idx;
	float	max_score;
	float	sum_exp;

	max_score = -INFINITY;
	sum_exp = 0.0f;
	memset(out, 0, head_dim * sizeof(float));
	i = 0;
	while (i < n_selected)
	{
		phys_idx = selected_blocks[i];
		if (phys_idx >= 0)
		{
			attention_block(out, q, &bm->blocks[phys_idx],
				kv_h, head_dim, scale, &max_score, &sum_exp);
		}
		i++;
	}
	if (sum_exp > 0.0f)
		simd_scale_f32(out, 1.0f / sum_exp, head_dim);
}

/*
** Multi-head paged attention with GQA support
*/
void	op_paged_attention(float *output, const float *q,
			t_paged_kv_cache *pkv, int n_heads, int n_kv_heads,
			int head_dim, float scale)
{
	int	h;
	int	kv_h;

	memset(output, 0, n_heads * head_dim * sizeof(float));
	#pragma omp parallel for schedule(static) private(kv_h)
	for (h = 0; h < n_heads; h++)
	{
		kv_h = h / (n_heads / n_kv_heads);
		attention_head_paged(
			output + h * head_dim,
			q + h * head_dim,
			pkv, kv_h, head_dim, scale);
	}
}

/*
** Multi-head paged sparse attention
*/
void	op_paged_attention_sparse(float *output, const float *q,
			t_block_manager *bm, const int *selected_blocks, int n_selected,
			int n_heads, int n_kv_heads, int head_dim, float scale)
{
	int	h;
	int	kv_h;

	memset(output, 0, n_heads * head_dim * sizeof(float));
	#pragma omp parallel for schedule(static) private(kv_h)
	for (h = 0; h < n_heads; h++)
	{
		kv_h = h / (n_heads / n_kv_heads);
		attention_head_paged_sparse(
			output + h * head_dim,
			q + h * head_dim,
			bm, selected_blocks, n_selected,
			kv_h, head_dim, scale);
	}
}
