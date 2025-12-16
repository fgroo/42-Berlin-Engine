/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   kv_cache_score.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "kv_cache_internal.h"
#include <string.h>

static float	head_dot(t_evict_ctx *ctx, const t_tensor *q, int h, int s)
{
	float	dot;
	int		d;
	int		off;
	t_bf16	*q_data;
	t_bf16	*k_data;

	q_data = (t_bf16 *)q->data;
	k_data = (t_bf16 *)ctx->cache->k.data;
	dot = 0.0f;
	d = 0;
	off = (s * ctx->heads + h) * ctx->dim;
	while (d < ctx->dim)
	{
		dot += bf16_to_float(q_data[h * ctx->dim + d])
			* bf16_to_float(k_data[off + d]);
		d++;
	}
	if (dot < 0)
		dot = 0.0f;
	return (dot);
}

static void	score_one(t_evict_ctx *ctx, const t_tensor *q,
				const t_tensor *w, int s)
{
	float	total;
	int		h;
	t_bf16	*w_data;

	w_data = (t_bf16 *)w->data;
	total = 0.0f;
	h = 0;
	while (h < ctx->heads)
	{
		total += bf16_to_float(w_data[h]) * head_dot(ctx, q, h, s);
		h++;
	}
	ctx->si[s].val = total;
	ctx->si[s].idx = s;
}

void	compute_evict_scores(t_evict_ctx *ctx, const t_tensor *q,
			const t_tensor *w)
{
	int	s;

	s = 0;
	while (s < ctx->n)
	{
		score_one(ctx, q, w, s);
		s++;
	}
}

void	compact_kv_cache(t_evict_ctx *ctx, int keep_k)
{
	int	i;
	int	old_idx;
	int	blk;

	blk = ctx->heads * ctx->dim;
	i = 0;
	while (i < keep_k)
	{
		old_idx = ctx->si[i].idx;
		if (i != old_idx)
		{
			memcpy((t_bf16 *)ctx->cache->k.data + i * blk,
				(t_bf16 *)ctx->cache->k.data + old_idx * blk,
				blk * sizeof(t_bf16));
			memcpy((t_bf16 *)ctx->cache->v.data + i * blk,
				(t_bf16 *)ctx->cache->v.data + old_idx * blk,
				blk * sizeof(t_bf16));
		}
		i++;
	}
}
