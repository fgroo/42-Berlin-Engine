/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_topk.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/08 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/*
** QuickSelect with Median-of-3 pivot selection
** O(N) average case for selecting Top-K elements
** Avoids O(N²) worst case on sorted/reverse-sorted arrays
*/

static void	swap_score_idx(t_score_idx *a, t_score_idx *b)
{
	t_score_idx	tmp;

	tmp = *a;
	*a = *b;
	*b = tmp;
}

/*
** Median-of-3 pivot: choose median of first, middle, last elements
** This avoids O(N²) worst case when array is already sorted
*/
static int	median_of_three(t_score_idx *arr, int lo, int hi)
{
	int	mid;

	mid = lo + (hi - lo) / 2;
	if (arr[lo].val < arr[mid].val)
		swap_score_idx(&arr[lo], &arr[mid]);
	if (arr[lo].val < arr[hi].val)
		swap_score_idx(&arr[lo], &arr[hi]);
	if (arr[mid].val < arr[hi].val)
		swap_score_idx(&arr[mid], &arr[hi]);
	swap_score_idx(&arr[mid], &arr[hi - 1]);
	return (hi - 1);
}

/*
** Partition for descending order (larger values first)
** Returns final position of pivot
*/
static int	partition_desc(t_score_idx *arr, int lo, int hi)
{
	int		pivot_idx;
	float	pivot;
	int		i;
	int		j;

	if (hi - lo < 3)
		pivot_idx = hi;
	else
		pivot_idx = median_of_three(arr, lo, hi);
	pivot = arr[pivot_idx].val;
	swap_score_idx(&arr[pivot_idx], &arr[hi]);
	i = lo - 1;
	j = lo;
	while (j < hi)
	{
		if (arr[j].val >= pivot)
		{
			i++;
			swap_score_idx(&arr[i], &arr[j]);
		}
		j++;
	}
	swap_score_idx(&arr[i + 1], &arr[hi]);
	return (i + 1);
}

/*
** QuickSelect: Rearrange array so top K elements are at indices [0..k-1]
** Elements within top K are not necessarily sorted
** O(N) average, O(N²) worst case (mitigated by median-of-3)
*/
static void	quickselect_k(t_score_idx *arr, int n, int k)
{
	int	lo;
	int	hi;
	int	p;

	lo = 0;
	hi = n - 1;
	while (lo < hi)
	{
		p = partition_desc(arr, lo, hi);
		if (p == k - 1)
			return ;
		if (p < k - 1)
			lo = p + 1;
		else
			hi = p - 1;
	}
}

static void	fill_temp_f32(t_score_idx *temp, float *data, int n)
{
	int	i;

	i = 0;
	while (i < n)
	{
		temp[i].val = data[i];
		temp[i].idx = i;
		i++;
	}
}

static void	fill_temp_bf16(t_score_idx *temp, t_bf16 *data, int n)
{
	int	i;

	i = 0;
	while (i < n)
	{
		temp[i].val = bf16_to_float(data[i]);
		temp[i].idx = i;
		i++;
	}
}

void	op_topk_select(int *indices, const t_tensor *scores,
			int k, t_arena *scratch)
{
	int			n;
	size_t		saved;
	t_score_idx	*temp;
	int			i;

	n = scores->size;
	if (k > n)
		k = n;
	saved = scratch->offset;
	temp = arena_alloc_or_die(scratch, n * sizeof(t_score_idx));
	if (scores->dtype == DTYPE_F32)
		fill_temp_f32(temp, (float *)scores->data, n);
	else
		fill_temp_bf16(temp, (t_bf16 *)scores->data, n);
	quickselect_k(temp, n, k);
	i = 0;
	while (i < k)
	{
		indices[i] = temp[i].idx;
		i++;
	}
	scratch->offset = saved;
}
