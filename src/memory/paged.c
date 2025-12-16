/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   paged.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 19:20:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/14 19:20:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "paged.h"
#include "compute/ops_heap.h"
#include "compute/simd_kernels.h"  /* simd_dot_f32_f32 for block scoring */
#include "../safe_alloc.h"  /* xmalloc/xcalloc for safe allocation (Phase 9) */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
** ============================================================================
** Block Manager Implementation
** ============================================================================
*/

int	block_manager_init(t_block_manager *bm, int n_layers, int n_kv_heads,
			int head_dim, int max_blocks_per_layer)
{
	int		total_blocks;
	int		block_size_bytes;
	int		i;

	bm->n_layers = n_layers;
	bm->n_kv_heads = n_kv_heads;
	bm->head_dim = head_dim;
	bm->max_logical = max_blocks_per_layer;
	bm->blocks = NULL;
	bm->page_table = NULL;
	bm->free_stack = NULL;
	bm->logical_len = NULL;
	
	/* Total physical blocks = layers * max_per_layer */
	total_blocks = n_layers * max_blocks_per_layer;
	bm->capacity = total_blocks;
	
	/* Allocate block pool */
	bm->blocks = calloc(total_blocks, sizeof(t_kv_block));
	if (!bm->blocks)
	{
		fprintf(stderr, "[PagedKV] ERROR: Failed to allocate block pool\n");
		return (-1);
	}
	
	/* Size of K or V per block: [n_kv_heads * BLOCK_SIZE * head_dim] */
	block_size_bytes = n_kv_heads * KV_BLOCK_SIZE * head_dim * sizeof(t_bf16);
	
	/* Allocate K/V storage for each block */
	i = 0;
	while (i < total_blocks)
	{
		bm->blocks[i].k = malloc(block_size_bytes);
		bm->blocks[i].v = malloc(block_size_bytes);
		bm->blocks[i].centroid = calloc(head_dim, sizeof(float));
		if (!bm->blocks[i].k || !bm->blocks[i].v || !bm->blocks[i].centroid)
		{
			fprintf(stderr, "[PagedKV] ERROR: Failed to allocate block %d\n", i);
			block_manager_free(bm);
			return (-1);
		}
		memset(bm->blocks[i].k, 0, block_size_bytes);
		memset(bm->blocks[i].v, 0, block_size_bytes);
		bm->blocks[i].n_tokens = 0;
		bm->blocks[i].start_pos = -1;
		bm->blocks[i].layer_idx = -1;
		bm->blocks[i].lsh_registered = 0;
		bm->blocks[i].lsh_hash = 0;
		i++;
	}
	
	/* Allocate page table: [n_layers][max_logical] */
	bm->page_table = calloc(n_layers * max_blocks_per_layer, sizeof(int));
	if (!bm->page_table)
	{
		fprintf(stderr, "[PagedKV] ERROR: Failed to allocate page table\n");
		block_manager_free(bm);
		return (-1);
	}
	/* Initialize page table to -1 (unmapped) */
	i = 0;
	while (i < n_layers * max_blocks_per_layer)
	{
		bm->page_table[i] = -1;
		i++;
	}
	
	/* Allocate logical length counter per layer */
	bm->logical_len = calloc(n_layers, sizeof(int));
	
	/* Initialize free stack with all blocks */
	bm->free_stack = malloc(total_blocks * sizeof(int));
	if (!bm->free_stack)
	{
		fprintf(stderr, "[PagedKV] ERROR: Failed to allocate free stack\n");
		block_manager_free(bm);
		return (-1);
	}
	i = 0;
	while (i < total_blocks)
	{
		bm->free_stack[i] = i;
		i++;
	}
	bm->free_top = total_blocks - 1;
	
	printf("[PagedKV] Initialized: %d blocks, %d layers, %d KV heads, %d head_dim\n",
		total_blocks, n_layers, n_kv_heads, head_dim);
	return (0);
}

void	block_manager_free(t_block_manager *bm)
{
	int	i;

	if (!bm)
		return ;
	if (bm->blocks)
	{
		i = 0;
		while (i < bm->capacity)
		{
			free(bm->blocks[i].k);
			free(bm->blocks[i].v);
			free(bm->blocks[i].centroid);
			i++;
		}
		free(bm->blocks);
	}
	free(bm->page_table);
	free(bm->free_stack);
	free(bm->logical_len);
	memset(bm, 0, sizeof(t_block_manager));
}

int	block_alloc(t_block_manager *bm)
{
	int	phys_idx;

	if (bm->free_top < 0)
	{
		fprintf(stderr, "[PagedKV] ERROR: Block pool exhausted!\n");
		return (-1);
	}
	phys_idx = bm->free_stack[bm->free_top];
	bm->free_top--;
	
	/* Reset block state */
	bm->blocks[phys_idx].n_tokens = 0;
	bm->blocks[phys_idx].start_pos = -1;
	bm->blocks[phys_idx].layer_idx = -1;
	bm->blocks[phys_idx].lsh_registered = 0;
	bm->blocks[phys_idx].lsh_hash = 0;
	
	return (phys_idx);
}

void	block_free(t_block_manager *bm, int phys_idx)
{
	if (phys_idx < 0 || phys_idx >= bm->capacity)
		return ;
	bm->free_top++;
	bm->free_stack[bm->free_top] = phys_idx;
}

/*
** ============================================================================
** Paged KV Cache Operations
** ============================================================================
*/

void	paged_kv_append(t_paged_kv_cache *pkv, const float *k, const float *v,
			int pos)
{
	t_block_manager	*bm;
	int				block_idx;		/* Logical block index */
	int				phys_idx;		/* Physical block index */
	int				offset_in_block;
	int				page_table_base;
	t_kv_block		*block;
	int				kv_size;
	int				h;
	int				d;

	bm = pkv->bm;
	block_idx = pos / KV_BLOCK_SIZE;
	offset_in_block = pos % KV_BLOCK_SIZE;
	page_table_base = pkv->layer_idx * bm->max_logical;
	
	/* Check if we need to allocate a new block */
	if (block_idx >= bm->logical_len[pkv->layer_idx])
	{
		/* Allocate new physical block */
		phys_idx = block_alloc(bm);
		if (phys_idx < 0)
		{
			fprintf(stderr, "[PagedKV] FATAL: Block pool exhausted at pos=%d\n", pos);
			exit(1);
		}
		
		/* Map logical -> physical */
		bm->page_table[page_table_base + block_idx] = phys_idx;
		bm->logical_len[pkv->layer_idx] = block_idx + 1;
		bm->blocks[phys_idx].start_pos = block_idx * KV_BLOCK_SIZE;
		pkv->n_blocks = block_idx + 1;
	}
	
	/* Get physical block */
	phys_idx = bm->page_table[page_table_base + block_idx];
	block = &bm->blocks[phys_idx];
	
	/* Write K/V into block at offset */
	/* Layout: [kv_head][token_in_block][dim] */
	kv_size = bm->head_dim;
	h = 0;
	while (h < bm->n_kv_heads)
	{
		/* Destination offset: h * BLOCK_SIZE * head_dim + offset * head_dim */
		int dest_offset = h * KV_BLOCK_SIZE * kv_size + offset_in_block * kv_size;
		/* Source offset: h * head_dim */
		int src_offset = h * kv_size;
		
		/* Convert F32 -> BF16 and store */
		d = 0;
		while (d < kv_size)
		{
			block->k[dest_offset + d] = float_to_bf16(k[src_offset + d]);
			block->v[dest_offset + d] = float_to_bf16(v[src_offset + d]);
			d++;
		}
		h++;
	}
	
	block->n_tokens = offset_in_block + 1;
	block->layer_idx = pkv->layer_idx;
	pkv->n_tokens = pos + 1;
	
	/* Calculate centroid when block is full (for sparse routing) */
	if (block->n_tokens == KV_BLOCK_SIZE && !block->lsh_registered)
	{
		calc_block_centroid(block, bm->head_dim);
		block->lsh_registered = 1;
	}
}

void	paged_kv_get(t_paged_kv_cache *pkv, int pos, int kv_head,
			t_bf16 **out_k, t_bf16 **out_v)
{
	t_block_manager	*bm;
	int				block_idx;
	int				offset_in_block;
	int				phys_idx;
	int				page_table_base;
	int				data_offset;

	bm = pkv->bm;
	block_idx = pos / KV_BLOCK_SIZE;
	offset_in_block = pos % KV_BLOCK_SIZE;
	page_table_base = pkv->layer_idx * bm->max_logical;
	
	phys_idx = bm->page_table[page_table_base + block_idx];
	if (phys_idx < 0)
	{
		*out_k = NULL;
		*out_v = NULL;
		return ;
	}
	
	/* Offset: [kv_head][token_in_block][0] */
	data_offset = kv_head * KV_BLOCK_SIZE * bm->head_dim + 
				  offset_in_block * bm->head_dim;
	
	*out_k = &bm->blocks[phys_idx].k[data_offset];
	*out_v = &bm->blocks[phys_idx].v[data_offset];
}

void	paged_kv_reset(t_paged_kv_cache *pkv)
{
	t_block_manager	*bm;
	int				page_table_base;
	int				i;
	int				phys_idx;

	bm = pkv->bm;
	page_table_base = pkv->layer_idx * bm->max_logical;
	
	/* Return all blocks to free pool */
	i = 0;
	while (i < bm->logical_len[pkv->layer_idx])
	{
		phys_idx = bm->page_table[page_table_base + i];
		if (phys_idx >= 0)
			block_free(bm, phys_idx);
		bm->page_table[page_table_base + i] = -1;
		i++;
	}
	
	bm->logical_len[pkv->layer_idx] = 0;
	pkv->n_blocks = 0;
	pkv->n_tokens = 0;
}

/*
** ============================================================================
** Block Centroid Calculation (Mean Pooling of Keys)
** ============================================================================
*/

void	calc_block_centroid(t_kv_block *block, int head_dim)
{
	int		t;
	int		d;
	float	inv_n;
	t_bf16	*k_ptr;

	if (block->n_tokens <= 0)
		return ;
	/* Zero the centroid */
	d = 0;
	while (d < head_dim)
	{
		block->centroid[d] = 0.0f;
		d++;
	}
	/* Sum all keys (from first KV head only - they share representation) */
	/* Block layout: [kv_head][token][dim] - we use head 0 */
	t = 0;
	while (t < block->n_tokens)
	{
		k_ptr = block->k + t * head_dim;  /* head 0 at offset 0 */
		d = 0;
		while (d < head_dim)
		{
			block->centroid[d] += bf16_to_float(k_ptr[d]);
			d++;
		}
		t++;
	}
	/* Divide by n_tokens to get mean */
	inv_n = 1.0f / (float)block->n_tokens;
	d = 0;
	while (d < head_dim)
	{
		block->centroid[d] *= inv_n;
		d++;
	}
}

/*
** ============================================================================
** Block Scoring for Sparse Attention (Q · Centroid Similarity)
** ============================================================================
*/

void	score_blocks(t_block_score *scores, const float *q,
			t_paged_kv_cache *pkv, int n_blocks)
{
	t_block_manager	*bm;
	int				page_base;
	int				i;
	int				phys_idx;
	float			dot;
	t_kv_block		*block;

	bm = pkv->bm;
	page_base = pkv->layer_idx * bm->max_logical;
	i = 0;
	while (i < n_blocks)
	{
		phys_idx = bm->page_table[page_base + i];
		scores[i].block_idx = i;
		scores[i].phys_idx = phys_idx;
		if (phys_idx < 0)
		{
			scores[i].score = -1e9f;  /* Invalid block */
			i++;
			continue ;
		}
		block = &bm->blocks[phys_idx];
		/* SIMD optimized: Q · centroid (8x faster than scalar) */
		dot = simd_dot_f32_f32(q, block->centroid, bm->head_dim);
		scores[i].score = dot;
		i++;
	}
}

/*
** Select top-K blocks using MIN-HEAP for O(N log K) complexity
** ALWAYS includes:
** - Block 0 (prefix anchor - system prompt, initial context)
** - Last block (suffix anchor - local window for recency)
*/
int	select_top_k_blocks(int *selected, t_block_score *scores,
			int n_blocks, int k)
{
	int			selected_count;
	int			first_phys;
	int			last_phys;
	int			actual_k;
	int			heap_size;
	int			i;
	t_heap_item	heap[64];  /* Max K = 64, stack safe */

	if (n_blocks <= 0)
		return (0);
	
	/* Always include FIRST block (prefix anchor for system/initial context) */
	first_phys = scores[0].phys_idx;
	selected[0] = first_phys;
	selected_count = 1;
	
	/* Always include LAST block (suffix anchor for local window) */
	last_phys = scores[n_blocks - 1].phys_idx;
	if (last_phys != first_phys && n_blocks > 1)
	{
		selected[selected_count] = last_phys;
		selected_count++;
	}
	
	if (n_blocks <= 2 || k <= 2)
		return (selected_count);
	
	/* Remaining slots for top-scoring middle blocks */
	actual_k = k - 2;
	if (actual_k > n_blocks - 2)
		actual_k = n_blocks - 2;
	if (actual_k > 62)
		actual_k = 62;  /* Cap to heap array size */
	
	/* BUILD MIN-HEAP: O(N log K) selection of middle blocks */
	/* Push scores[1..n_blocks-2] into heap, skip anchored first/last */
	heap_size = 0;
	i = 1;
	while (i < n_blocks - 1)
	{
		heap_push(heap, &heap_size, actual_k, scores[i].score, i);
		i++;
	}
	
	/* Extract winning indices from heap */
	i = 0;
	while (i < heap_size)
	{
		selected[selected_count] = scores[heap[i].index].phys_idx;
		selected_count++;
		i++;
	}
	return (selected_count);
}
