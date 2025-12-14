/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_rope.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/08 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include <math.h>
#include <stdio.h>

/*
** THREAD-SAFETY FIX: Removed static g_thetas[256] global buffer.
** With OpenMP parallelization, multiple threads writing to static buffer
** causes race conditions and corrupted rotations ("model amnesia").
** Now uses stack-local buffer passed as parameter - thread-safe by design.
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
	
	// Use cached thetas if provided, otherwise compute on the fly
	if (ctx.thetas_cache)
	{
		thetas_ptr = ctx.thetas_cache;
	}
	else
	{
		// Fallback: compute thetas (slow path with pow() calls)
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
			rope_apply_f32((float *)x->data + i * ctx.head_dim, &ctx, thetas_ptr);
		else if (x->dtype == DTYPE_BF16)
			rope_apply_bf16((t_bf16 *)x->data + i * ctx.head_dim, &ctx, thetas_ptr);
		i++;
	}
}
