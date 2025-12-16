/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   sampler_topp_main.c                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "sampler_internal.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

static int	g_rng = 0;

void	compute_softmax(t_prob_idx *pi, const t_tensor *log, float temp);

static float	rand_float(void)
{
	if (!g_rng)
	{
		srand((unsigned int)time(NULL));
		g_rng = 1;
	}
	return ((float)rand() / (float)RAND_MAX);
}

static int	cmp_prob(const void *a, const void *b)
{
	float	pa;
	float	pb;

	pa = ((const t_prob_idx *)a)->prob;
	pb = ((const t_prob_idx *)b)->prob;
	return ((pb > pa) - (pb < pa));
}

static int	sample_nucleus(t_prob_idx *pi, size_t cutoff, float top_sum)
{
	float	r;
	float	cum;
	size_t	i;

	r = rand_float() * top_sum;
	cum = 0.0f;
	i = 0;
	while (i < cutoff)
	{
		cum += pi[i].prob;
		if (r < cum)
			return (pi[i].idx);
		i++;
	}
	return (pi[0].idx);
}

int	sample_top_p(const t_tensor *logits, float temperature,
		float top_p, t_arena *scratch)
{
	size_t		saved;
	t_prob_idx	*pi;
	size_t		cutoff;
	float		cum;

	saved = scratch->offset;
	pi = arena_alloc_or_die(scratch, logits->size * sizeof(t_prob_idx));
	compute_softmax(pi, logits, temperature);
	qsort(pi, logits->size, sizeof(t_prob_idx), cmp_prob);
	cum = 0.0f;
	cutoff = 0;
	while (cutoff < logits->size && cum < top_p)
	{
		cum += pi[cutoff].prob;
		cutoff++;
	}
	cutoff = sample_nucleus(pi, cutoff, cum);
	scratch->offset = saved;
	return ((int)cutoff);
}
