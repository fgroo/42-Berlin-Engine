/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_lsh.h                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 12:00:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 12:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_LSH_H
# define OPS_LSH_H

# include <stdint.h>

/*
** LSH (Locality Sensitive Hashing) for Sparse Attention
** ======================================================
** Block-based indexing with lazy incremental updates.
** 
** Key insight: We DON'T hash every key. We hash the CENTROID (mean)
** of each block of 64 tokens. This reduces index overhead by 64x.
**
** Update cycle:
** - Token 0-63: Accumulate keys into sum_buffer
** - Token 64: Compute centroid, hash, insert into index, reset
** - Query: Find candidate blocks via hash, scan them + active block
*/

/* LSH Configuration */
# define LSH_NUM_HASHES 16      /* Bits in hash (16 = 65536 buckets) */
# define LSH_MAX_DIM 256        /* Max vector dimension */
# define LSH_BLOCK_SIZE 64      /* Tokens per KV block */
# define LSH_MAX_BLOCKS 4096    /* Max blocks (256k context) */
# define LSH_HAMMING_RADIUS 4   /* Hamming distance for candidates (was 2, too strict) */
# define LSH_NUM_BUCKETS 65536  /* 2^16 buckets */

/*
** LSH Context - random hyperplanes (initialized once at model load)
*/
typedef struct s_lsh_ctx
{
	float	hyperplanes[LSH_NUM_HASHES][LSH_MAX_DIM];
	int		dim;
	int		initialized;
}	t_lsh_ctx;

/*
** Block Accumulator - tracks current incomplete block
** Updated incrementally as each key is added to KV cache
*/
typedef struct s_block_acc
{
	float	sum_buffer[LSH_MAX_DIM];  /* Running sum of keys in block */
	int		count;                     /* Tokens in current block */
	int		start_pos;                 /* Starting position of block */
}	t_block_acc;

/*
** LSH Index - hash table mapping hash -> block IDs
** Uses open addressing with linear probing for cache locality
*/
typedef struct s_lsh_index
{
	/* Block metadata */
	uint16_t	block_hashes[LSH_MAX_BLOCKS];  /* Hash per block ID */
	int			block_starts[LSH_MAX_BLOCKS];  /* Start pos per block */
	int			block_lens[LSH_MAX_BLOCKS];    /* Tokens per block */
	int			n_blocks;                       /* Committed blocks */
	
	/* Hash table: bucket -> first block, chains for collisions */
	int			bucket_heads[LSH_NUM_BUCKETS]; /* -1 = empty */
	int			chain_next[LSH_MAX_BLOCKS];    /* -1 = end */
	
	/* Active block (not yet indexed) */
	t_block_acc	active;
	int			block_size;
}	t_lsh_index;

/* Initialize LSH with random hyperplanes */
void		lsh_init(t_lsh_ctx *ctx, int dim, unsigned int seed);

/* Initialize empty index */
void		lsh_index_init(t_lsh_index *idx);

/* Compute LSH hash for a vector (16-bit) */
uint16_t	lsh_hash(const t_lsh_ctx *ctx, const float *vec);

/* Hamming distance between two hashes */
int			lsh_hamming_distance(uint16_t h1, uint16_t h2);

/* Insert block into hash table */
void		lsh_insert_block(t_lsh_index *idx, uint16_t hash, 
				int start_pos, int len);

/* Update index with new key (called from kv_cache_append) */
void		lsh_update(t_lsh_index *idx, const t_lsh_ctx *ctx,
				const float *key, int dim, int pos);

/* Find candidate blocks (returns block count) */
int			lsh_find_candidates(const t_lsh_index *idx, uint16_t query_hash,
				int *block_ids, int max_blocks);

/* Get active block info (must always be scanned) */
int			lsh_get_active_block(const t_lsh_index *idx, 
				int *start_pos, int *len);

#endif
