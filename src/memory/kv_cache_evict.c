/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   kv_cache_evict.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/08 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "kv_cache.h"
#include "kv_cache_internal.h"
#include <stdlib.h>
#include <stdio.h>

static int	cmp_score_desc(const void *a, const void *b)
{
	float	va;
	float	vb;

	va = ((const t_score_index *)a)->val;
	vb = ((const t_score_index *)b)->val;
	return ((vb > va) - (vb < va));
}

static int	cmp_index_asc(const void *a, const void *b)
{
	return (((const t_score_index *)a)->idx
		- ((const t_score_index *)b)->idx);
}

void	kv_cache_evict(t_kv_cache *c, t_evict_params *p)
{
	size_t		saved;
	t_evict_ctx	ctx;
	int			keep_k;

	/* BOUNDS CHECK: Validate keep_k to prevent buffer overflow */
	if (p->keep_k <= 0)
	{
		fprintf(stderr, "Warning: kv_cache_evict keep_k=%d invalid\n", p->keep_k);
		return ;
	}
	keep_k = p->keep_k;
	/* Clamp keep_k to current sequence length (can't keep more than we have) */
	if (keep_k > c->current_seq_len)
		keep_k = c->current_seq_len;
	/* Clamp keep_k to max_seq_len (can't exceed cache capacity) */
	if (keep_k > c->max_seq_len)
		keep_k = c->max_seq_len;
	/* Early exit if nothing to evict */
	if (c->current_seq_len <= keep_k)
		return ;
	saved = p->scratch->offset;
	ctx.n = c->current_seq_len;
	ctx.heads = c->num_heads;
	ctx.dim = c->head_dim;
	ctx.cache = c;
	ctx.si = arena_alloc_or_die(p->scratch, ctx.n * sizeof(t_score_index));
	compute_evict_scores(&ctx, p->q, p->w);
	qsort(ctx.si, ctx.n, sizeof(t_score_index), cmp_score_desc);
	/* BOUNDS CHECK: Only sort keep_k elements (validated above) */
	qsort(ctx.si, keep_k, sizeof(t_score_index), cmp_index_asc);
	compact_kv_cache(&ctx, keep_k);
	c->current_seq_len = keep_k;
	p->scratch->offset = saved;
}
