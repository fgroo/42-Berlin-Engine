/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_heap.h                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 15:00:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/14 15:00:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_HEAP_H
# define OPS_HEAP_H

/*
** Min-Heap for O(n log k) Top-K selection.
** Maintains the K largest values seen so far.
** The heap root is the SMALLEST of the top-K (min-heap property).
** When a new value > root arrives, we replace root and sift down.
*/

typedef struct s_heap_item
{
	float	score;
	int		index;
}	t_heap_item;

/*
** Sift down: restore min-heap property after replacing root
** Time: O(log k)
*/
static inline void	heap_sift_down(t_heap_item *heap, int size, int i)
{
	int		smallest;
	int		left;
	int		right;
	t_heap_item	tmp;

	while (1)
	{
		smallest = i;
		left = 2 * i + 1;
		right = 2 * i + 2;
		if (left < size && heap[left].score < heap[smallest].score)
			smallest = left;
		if (right < size && heap[right].score < heap[smallest].score)
			smallest = right;
		if (smallest == i)
			break ;
		tmp = heap[i];
		heap[i] = heap[smallest];
		heap[smallest] = tmp;
		i = smallest;
	}
}

/*
** Sift up: restore min-heap property after insertion
** Time: O(log k)
*/
static inline void	heap_sift_up(t_heap_item *heap, int i)
{
	int			parent;
	t_heap_item	tmp;

	while (i > 0)
	{
		parent = (i - 1) / 2;
		if (heap[parent].score <= heap[i].score)
			break ;
		tmp = heap[i];
		heap[i] = heap[parent];
		heap[parent] = tmp;
		i = parent;
	}
}

/*
** Push item onto heap (maintaining size k)
** If heap is full and new score > root, replace root
** Time: O(log k)
*/
static inline void	heap_push(t_heap_item *heap, int *size, int capacity,
					float score, int index)
{
	if (*size < capacity)
	{
		heap[*size].score = score;
		heap[*size].index = index;
		heap_sift_up(heap, *size);
		(*size)++;
	}
	else if (score > heap[0].score)
	{
		heap[0].score = score;
		heap[0].index = index;
		heap_sift_down(heap, *size, 0);
	}
}

/*
** Build min-heap from unsorted array
** Time: O(k)
*/
static inline void	heap_build(t_heap_item *heap, int size)
{
	int	i;

	i = size / 2 - 1;
	while (i >= 0)
	{
		heap_sift_down(heap, size, i);
		i--;
	}
}

/*
** Extract indices from heap (unsorted order)
** Returns array of indices representing top-K items
*/
static inline void	heap_extract_indices(t_heap_item *heap, int size,
					int *out_indices)
{
	int	i;

	i = 0;
	while (i < size)
	{
		out_indices[i] = heap[i].index;
		i++;
	}
}

#endif
