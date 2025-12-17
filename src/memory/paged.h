/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   paged.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 19:20:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/14 19:20:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef PAGED_H
# define PAGED_H

# include "tensor/tensor.h"
# include "arena.h"
# include <stdint.h>

/*
** ============================================================================
** PAGED KV CACHE - DeepSeek V3.2 Sparse Attention Foundation
** ============================================================================
** Key insight: Contiguous [Seq, Head, Dim] layout causes cache thrashing.
** We use block-based paging:
** - KV cache is split into BLOCK_SIZE token chunks
** - Blocks are non-contiguous in memory (page table maps logical->physical)
** - Within each block: [n_kv_heads, BLOCK_SIZE, head_dim] for SIMD-friendly access
** - Enables O(K) sparse attention by selecting only relevant blocks
** ============================================================================
*/

/* DeepSeek V3.2 uses 16-token blocks for fine-grained sparse selection */
# ifndef KV_BLOCK_SIZE
#  define KV_BLOCK_SIZE 16
# endif

/* Maximum blocks per layer (4096 * 16 = 64K tokens max context) */
# ifndef KV_MAX_BLOCKS
#  define KV_MAX_BLOCKS 4096
# endif

/*
** Single KV block - holds BLOCK_SIZE consecutive tokens for one layer
** 
** PHASE 2 DEEP FREEZE: INT8 Quantized Storage
** - 2x memory reduction vs BF16
** - Per-token scaling for precision preservation
** - JIT dequantization on attention read
**
** Layout: [n_kv_heads][BLOCK_SIZE][head_dim] - Head-first for SIMD per-head access
*/
typedef struct s_kv_block
{
	/* INT8 Quantized Storage (2x smaller than BF16) */
	int8_t		*k_quant;		/* [n_kv_heads * BLOCK_SIZE * head_dim] */
	int8_t		*v_quant;		/* [n_kv_heads * BLOCK_SIZE * head_dim] */
	
	/* Per-token scaling factors for dequantization: x_real = x_quant * scale */
	float		*k_scales;		/* [n_kv_heads * BLOCK_SIZE] */
	float		*v_scales;		/* [n_kv_heads * BLOCK_SIZE] */
	
	/* Routing metadata (kept in FP32 for indexer precision) */
	float		*centroid;		/* Mean key vector for LSH [head_dim] */
	int			n_tokens;		/* How many tokens in this block (0-BLOCK_SIZE) */
	int			start_pos;		/* First token position in this block */
	int			layer_idx;		/* Which layer this block belongs to */
	int			lsh_registered;	/* 1 = centroid registered with LSH indexer */
	uint32_t	lsh_hash;		/* Cached LSH hash of centroid */
}	t_kv_block;

/*
** Block Manager - manages paged memory pool for all layers
** Replaces the old linear kv_cache array
*/
typedef struct s_block_manager
{
	t_kv_block	*blocks;		/* Pool of pre-allocated blocks */
	int			*page_table;	/* [n_layers][max_logical_blocks] -> physical idx */
	int			*free_stack;	/* Stack of free block indices */
	int			free_top;		/* Top of free stack (-1 = empty) */
	int			capacity;		/* Total physical blocks in pool */
	int			n_layers;		/* Number of layers */
	int			n_kv_heads;		/* KV heads per layer */
	int			head_dim;		/* Dimension per head */
	int			max_logical;	/* Max blocks per layer */
	int			*logical_len;	/* [n_layers] current logical block count */
}	t_block_manager;

/*
** Paged KV cache - per-layer view into block manager
*/
typedef struct s_paged_kv_cache
{
	t_block_manager	*bm;		/* Shared block manager */
	int				layer_idx;	/* Which layer this view is for */
	int				n_blocks;	/* Current number of blocks for this layer */
	int				n_tokens;	/* Total tokens in this layer's cache */
}	t_paged_kv_cache;

/* ============================================================================
** API Functions
** ============================================================================
*/

/*
** Initialize block manager with pre-allocated pool
** @param bm: Block manager to initialize
** @param n_layers: Number of transformer layers
** @param n_kv_heads: KV heads per layer
** @param head_dim: Dimension per head
** @param max_blocks_per_layer: Max blocks per layer (determines context length)
** @return: 0 on success, -1 on allocation failure
*/
int	block_manager_init(t_block_manager *bm, int n_layers, int n_kv_heads,
			int head_dim, int max_blocks_per_layer);

/*
** Free block manager and all allocated memory
*/
void	block_manager_free(t_block_manager *bm);

/*
** Allocate a new block from the pool
** @return: Physical block index, or -1 if pool exhausted
*/
int		block_alloc(t_block_manager *bm);

/*
** Return a block to the free pool
** @param phys_idx: Physical block index to free
*/
void	block_free(t_block_manager *bm, int phys_idx);

/*
** Append a single token's K/V to the paged cache
** Automatically allocates new block when current fills up
** @param pkv: Paged KV cache for specific layer
** @param k: Key vector [n_kv_heads * head_dim] in F32
** @param v: Value vector [n_kv_heads * head_dim] in F32
** @param pos: Global position of this token
*/
void	paged_kv_append(t_paged_kv_cache *pkv, const float *k, const float *v,
			int pos);

/*
** Get pointer to K/V for a specific position (INT8 quantized)
** @param pkv: Paged KV cache
** @param pos: Token position
** @param kv_head: Which KV head
** @param out_k: Output pointer to INT8 K data
** @param out_v: Output pointer to INT8 V data
** @param out_k_scale: Output K dequantization scale
** @param out_v_scale: Output V dequantization scale
*/
void	paged_kv_get_int8(t_paged_kv_cache *pkv, int pos, int kv_head,
			int8_t **out_k, int8_t **out_v, float *out_k_scale, float *out_v_scale);

/*
** Reset cache for a layer (start fresh context)
*/
void	paged_kv_reset(t_paged_kv_cache *pkv);

/*
** ============================================================================
** Block Centroids for Sparse Routing
** ============================================================================
*/

/*
** Calculate block centroid (mean of all key vectors in block) - INT8 version
** Dequantizes INT8 keys and computes mean for sparse routing
** Called when block is full (n_tokens == BLOCK_SIZE)
** @param block: Block to calculate centroid for
** @param head_dim: Dimension per head (uses first KV head)
** @param n_kv_heads: Number of KV heads (for layout calculation)
*/
void	calc_block_centroid_int8(t_kv_block *block, int head_dim, int n_kv_heads);

/*
** ============================================================================
** Block Scoring for Sparse Attention
** ============================================================================
*/

typedef struct s_block_score
{
	int		block_idx;	/* Logical block index */
	int		phys_idx;	/* Physical block index */
	float	score;		/* Importance score (Q·centroid) */
}	t_block_score;

/*
** Score blocks using centroid dot product with query
** @param scores: Output array [n_blocks]
** @param q: Query vector [head_dim] (single head)
** @param pkv: Paged KV cache
** @param n_blocks: Number of blocks to score
*/
void	score_blocks(t_block_score *scores, const float *q,
			t_paged_kv_cache *pkv, int n_blocks);

/*
** Select top-K most relevant blocks (plus mandatory tail for local window)
** @param selected: Output array of selected physical block indices
** @param scores: Input scores array
** @param n_blocks: Total blocks
** @param k: How many to select
** @return: Actual number selected
*/
int		select_top_k_blocks(int *selected, t_block_score *scores,
			int n_blocks, int k);

#endif
