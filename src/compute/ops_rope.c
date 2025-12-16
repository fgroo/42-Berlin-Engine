/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_rope.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/16 20:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include "ops_rope.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/*
** ============================================================================
** PRECOMPUTED ROPE CACHE (Phase 12: Eliminate 53K sinf/cosf per token)
** ============================================================================
** Before: sinf/cosf called 64 * 32 * 26 = 53,248 times per token
** After:  Just table lookups + FMA operations
**
** Memory cost: 8K * 64 * 2 * 4 = 4MB (excellent trade-off)
** ============================================================================
*/

t_rope_cache	*rope_cache_init(int head_dim, int max_seq_len, float theta_base)
{
	t_rope_cache	*cache;
	float			*inv_freq;
	int				half_dim;
	int				p;
	int				i;
	int				idx;
	float			theta;

	cache = malloc(sizeof(t_rope_cache));
	if (!cache)
		return (NULL);
	half_dim = head_dim / 2;
	cache->head_dim = head_dim;
	cache->max_seq_len = max_seq_len;
	/* 32-byte alignment for AVX2 SIMD loads */
	if (posix_memalign((void **)&cache->cos_table, 32,
			max_seq_len * half_dim * sizeof(float)) != 0)
	{
		free(cache);
		return (NULL);
	}
	if (posix_memalign((void **)&cache->sin_table, 32,
			max_seq_len * half_dim * sizeof(float)) != 0)
	{
		free(cache->cos_table);
		free(cache);
		return (NULL);
	}
	/* Precompute inverse frequencies: 1 / theta^(2i/d) */
	inv_freq = malloc(half_dim * sizeof(float));
	if (!inv_freq)
	{
		free(cache->cos_table);
		free(cache->sin_table);
		free(cache);
		return (NULL);
	}
	i = 0;
	while (i < half_dim)
	{
		inv_freq[i] = 1.0f / powf(theta_base, (float)(i * 2) / (float)head_dim);
		i++;
	}
	/* Fill tables: [Position][Frequency] */
	printf("[RoPE CACHE] Precomputing sin/cos for %d positions × %d dims...\n",
		max_seq_len, half_dim);
	p = 0;
	while (p < max_seq_len)
	{
		i = 0;
		while (i < half_dim)
		{
			theta = (float)p * inv_freq[i];
			idx = p * half_dim + i;
			cache->cos_table[idx] = cosf(theta);
			cache->sin_table[idx] = sinf(theta);
			i++;
		}
		p++;
	}
	free(inv_freq);
	printf("[RoPE CACHE] Done. Eliminated %d trig calls per token.\n",
		half_dim * 2);
	return (cache);
}

void	rope_cache_free(t_rope_cache *cache)
{
	if (!cache)
		return ;
	free(cache->cos_table);
	free(cache->sin_table);
	free(cache);
}

/*
** HOT PATH: Apply RoPE using precomputed tables.
** NO sinf/cosf calls - just table lookups and multiply-add.
** The 'restrict' keyword enables compiler autovectorization.
*/
void	rope_apply_cached(float *restrict q, float *restrict k,
			int pos, const t_rope_cache *restrict cache)
{
	int		half;
	int		i;
	float	*cos_row;
	float	*sin_row;
	float	c;
	float	s;
	float	q0;
	float	q1;
	float	k0;
	float	k1;

	if (pos >= cache->max_seq_len)
		return ;
	half = cache->head_dim / 2;
	cos_row = cache->cos_table + (pos * half);
	sin_row = cache->sin_table + (pos * half);
	/* Compiler can autovectorize this loop with -O3 -march=native */
	i = 0;
	while (i < half)
	{
		c = cos_row[i];
		s = sin_row[i];
		q0 = q[i];
		q1 = q[i + half];
		k0 = k[i];
		k1 = k[i + half];
		q[i] = q0 * c - q1 * s;
		q[i + half] = q0 * s + q1 * c;
		k[i] = k0 * c - k1 * s;
		k[i + half] = k0 * s + k1 * c;
		i++;
	}
}

/*
** Apply RoPE to a single vector (for batched prefill where Q/K are separate).
*/
void	rope_apply_single_cached(float *restrict vec, int pos,
			const t_rope_cache *restrict cache)
{
	int		half;
	int		i;
	float	*cos_row;
	float	*sin_row;
	float	c;
	float	s;
	float	v0;
	float	v1;

	if (pos >= cache->max_seq_len)
		return ;
	half = cache->head_dim / 2;
	cos_row = cache->cos_table + (pos * half);
	sin_row = cache->sin_table + (pos * half);
	i = 0;
	while (i < half)
	{
		c = cos_row[i];
		s = sin_row[i];
		v0 = vec[i];
		v1 = vec[i + half];
		vec[i] = v0 * c - v1 * s;
		vec[i + half] = v0 * s + v1 * c;
		i++;
	}
}

/*
** Apply RoPE to multiple heads at once (FAST PATH).
** Processes n_heads contiguous head vectors in one function call.
** Each head is [head_dim] floats, stored contiguously.
**
** @param data:    Pointer to first head [n_heads * head_dim]
** @param n_heads: Number of heads to process
** @param pos:     Position in sequence
** @param cache:   Precomputed sin/cos tables
*/
void	rope_apply_multihead_cached(float *restrict data, int n_heads, int pos,
			const t_rope_cache *restrict cache)
{
	int		half;
	int		head_dim;
	int		h;
	int		i;
	float	*cos_row;
	float	*sin_row;
	float	*head;
	float	c;
	float	s;
	float	v0;
	float	v1;

	if (pos >= cache->max_seq_len)
		return ;
	head_dim = cache->head_dim;
	half = head_dim / 2;
	cos_row = cache->cos_table + (pos * half);
	sin_row = cache->sin_table + (pos * half);
	/* Process all heads with same sin/cos row (same position) */
	h = 0;
	while (h < n_heads)
	{
		head = data + h * head_dim;
		i = 0;
		while (i < half)
		{
			c = cos_row[i];
			s = sin_row[i];
			v0 = head[i];
			v1 = head[i + half];
			head[i] = v0 * c - v1 * s;
			head[i + half] = v0 * s + v1 * c;
			i++;
		}
		h++;
	}
}

/*
** ============================================================================
** LEGACY API (Backward Compatibility)
** ============================================================================
** Kept for code that still uses op_rope() with t_rope_ctx.
** New code should use rope_apply_cached() directly.
** ============================================================================
*/

static float	compute_yarn_ramp(float freq_idx, const t_rope_ctx *ctx)
{
	float	start;
	float	end;
	float	alpha;
	double	theta_d;
	double	theta_scaled;

	start = ctx->beta_slow / ctx->head_dim;
	end = ctx->beta_fast / ctx->head_dim;
	alpha = (freq_idx - start) / (end - start);
	theta_d = 1.0 / pow(ctx->theta_base, freq_idx);
	theta_scaled = 1.0 / pow(ctx->theta_base * ctx->factor, freq_idx);
	return ((float)((1.0 - alpha) * theta_d + alpha * theta_scaled));
}

static float	get_yarn_theta(int j, const t_rope_ctx *ctx)
{
	float	freq_idx;
	double	theta_d;

	freq_idx = (float)j / (float)ctx->head_dim;
	theta_d = 1.0 / pow(ctx->theta_base, freq_idx);
	if (ctx->factor <= 1.0f)
		return ((float)theta_d);
	if (freq_idx < ctx->beta_slow / ctx->head_dim)
		return ((float)theta_d);
	if (freq_idx > ctx->beta_fast / ctx->head_dim)
		return ((float)(1.0 / pow(ctx->theta_base * ctx->factor, freq_idx)));
	return (compute_yarn_ramp(freq_idx, ctx));
}

static void	rope_apply_f32(float *vec, t_rope_ctx *ctx, const float *thetas)
{
	int		j;
	int		half;
	float	angle;
	float	v0;
	float	v1;

	half = ctx->head_dim / 2;
	j = 0;
	while (j < half)
	{
		angle = ctx->pos * thetas[j];
		v0 = vec[j];
		v1 = vec[j + half];
		if (ctx->mscale != 1.0f)
		{
			v0 *= ctx->mscale;
			v1 *= ctx->mscale;
		}
		vec[j] = v0 * cosf(angle) - v1 * sinf(angle);
		vec[j + half] = v0 * sinf(angle) + v1 * cosf(angle);
		j++;
	}
}

static void	rope_apply_bf16(t_bf16 *vec, t_rope_ctx *ctx, const float *thetas)
{
	int		j;
	int		half;
	float	angle;
	float	v0;
	float	v1;

	half = ctx->head_dim / 2;
	j = 0;
	while (j < half)
	{
		angle = ctx->pos * thetas[j];
		v0 = bf16_to_float(vec[j]);
		v1 = bf16_to_float(vec[j + half]);
		if (ctx->mscale != 1.0f)
		{
			v0 *= ctx->mscale;
			v1 *= ctx->mscale;
		}
		vec[j] = float_to_bf16(v0 * cosf(angle) - v1 * sinf(angle));
		vec[j + half] = float_to_bf16(v0 * sinf(angle) + v1 * cosf(angle));
		j++;
	}
}

void	op_rope(t_tensor *x, int pos, const t_rope_ctx *ctx_in)
{
	t_rope_ctx	ctx;
	float		thetas[256];
	float		*thetas_ptr;
	int			num_vecs;
	int			half;
	int			i;
	int			j;

	if (!x || !x->data)
		return ;
	ctx = *ctx_in;
	ctx.pos = pos;
	half = ctx.head_dim / 2;
	num_vecs = x->size / ctx.head_dim;
	/* Use cached thetas if provided, otherwise compute on the fly */
	if (ctx.thetas_cache)
	{
		thetas_ptr = ctx.thetas_cache;
	}
	else
	{
		/* Fallback: compute thetas (slow path with pow() calls) */
		j = 0;
		while (j < half)
		{
			thetas[j] = get_yarn_theta(j * 2, &ctx);
			j++;
		}
		thetas_ptr = thetas;
	}
	i = 0;
	while (i < num_vecs)
	{
		if (x->dtype == DTYPE_F32)
			rope_apply_f32((float *)x->data + i * ctx.head_dim, &ctx,
				thetas_ptr);
		else if (x->dtype == DTYPE_BF16)
			rope_apply_bf16((t_bf16 *)x->data + i * ctx.head_dim, &ctx,
				thetas_ptr);
		i++;
	}
}
