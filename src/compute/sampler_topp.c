/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   sampler_topp.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "sampler_internal.h"
#include <math.h>
#include <stdlib.h>

static float	find_max(const t_tensor *log)
{
	float	max_l;
	size_t	i;

	if (log->dtype == DTYPE_F32)
	{
		float *data = (float *)log->data;
		max_l = data[0];
		i = 1;
		while (i < log->size)
		{
			if (data[i] > max_l)
				max_l = data[i];
			i++;
		}
	}
	else
	{
		t_bf16 *data = (t_bf16 *)log->data;
		max_l = bf16_to_float(data[0]);
		i = 1;
		while (i < log->size)
		{
			if (bf16_to_float(data[i]) > max_l)
				max_l = bf16_to_float(data[i]);
			i++;
		}
	}
	return (max_l);
}

static float	exp_sum(t_prob_idx *pi, const t_tensor *log, float t, float m)
{
	float	sum;
	size_t	i;

	sum = 0.0f;
	i = 0;
	
	if (log->dtype == DTYPE_F32)
	{
		float *data = (float *)log->data;
		while (i < log->size)
		{
			pi[i].prob = expf((data[i] - m) / t);
			pi[i].idx = i;
			sum += pi[i].prob;
			i++;
		}
	}
	else
	{
		t_bf16 *data = (t_bf16 *)log->data;
		while (i < log->size)
		{
			pi[i].prob = expf((bf16_to_float(data[i]) - m) / t);
			pi[i].idx = i;
			sum += pi[i].prob;
			i++;
		}
	}
	return (sum);
}

static void	normalize(t_prob_idx *pi, size_t size, float sum)
{
	size_t	i;

	i = 0;
	while (i < size)
	{
		pi[i].prob /= sum;
		i++;
	}
}

void	compute_softmax(t_prob_idx *pi, const t_tensor *log, float temp)
{
	float	max_l;
	float	sum;

	max_l = find_max(log);
	sum = exp_sum(pi, log, temp, max_l);
	normalize(pi, log->size, sum);
}

static int	compare_prob(const void *a, const void *b)
{
	const t_prob_idx *pa = (const t_prob_idx *)a;
	const t_prob_idx *pb = (const t_prob_idx *)b;
	if (pa->prob > pb->prob) return (-1);
	if (pa->prob < pb->prob) return (1);
	return (0);
}

int	sample_top_p(const t_tensor *logits, float temperature, float top_p, t_arena *scratch)
{
	t_prob_idx	*pi;
	float		cum_prob;
	float		r;
	size_t		i;

	pi = arena_alloc(scratch, logits->size * sizeof(t_prob_idx));
	compute_softmax(pi, logits, temperature);
	
	// Sort by probability descending
	qsort(pi, logits->size, sizeof(t_prob_idx), compare_prob);
	
	// Cumulative sum
	cum_prob = 0.0f;
	i = 0;
	while (i < logits->size)
	{
		cum_prob += pi[i].prob;
		if (cum_prob > top_p)
		{
			// Include this token and stop
			i++;
			break;
		}
		i++;
	}
	
	// Sample from top-p set
	// Re-normalize probabilities within the set
	// Or just sample uniformly from [0, cum_prob]?
	// Standard Top-P: Sample r in [0, 1], find first where cum > r?
	// No, we need to sample from the truncated distribution.
	// Sample r in [0, cum_prob_cutoff]
	
	r = (float)rand() / (float)RAND_MAX * cum_prob;
	
	float running = 0.0f;
	for (size_t j = 0; j < i; j++)
	{
		running += pi[j].prob;
		if (running > r)
			return (pi[j].idx);
	}
	return (pi[i - 1].idx);
}
