/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_lsh.h                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 12:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/16 00:00:00 by fgroo            ###   ########.fr       */
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
** 
** Phase 10 TURBO: Added hp_transposed for column-major SIMD access.
** Original: hyperplanes[16][dim] - 16 separate dot products
** Transposed: hp_transposed[dim][16] - single pass with 16-wide FMA
*/
typedef struct s_lsh_ctx
{
	float	hyperplanes[LSH_NUM_HASHES][LSH_MAX_DIM];
	float	hp_transposed[LSH_MAX_DIM][LSH_NUM_HASHES];  /* TURBO: [dim][16] */
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

/*
** ===========================================================================
** LSH DIAGNOSTICS (Phase 9: Thread-Safe Stats)
** ===========================================================================
** Runtime monitoring of LSH quality to detect hash collision degradation.
** If recall drops below 80%, Nested Learning will train on corrupted data.
**
** CRITICAL FIX: Use atomics instead of __thread to aggregate stats
** correctly across OpenMP worker threads. Thread-local storage dies
** with the worker thread, losing all accumulated data.
*/

# include <stdatomic.h>

/*
** ATOMIC statistics for thread-safe aggregation across OpenMP threads
** All counters use atomic_fetch_add for lock-free updates.
** Floats are accumulated as scaled integers (x100) then converted.
*/
typedef struct s_lsh_stats_atomic
{
	atomic_uint_fast64_t	total_queries;       /* Total sparse attention queries */
	atomic_uint_fast64_t	validated_queries;   /* Queries with brute-force validation */
	atomic_uint_fast64_t	topk_hits;           /* Sum of (LSH ∩ BruteForce) tokens */
	atomic_uint_fast64_t	topk_total;          /* Sum of K across validated queries */
	atomic_uint_fast64_t	recall_scaled;       /* Sum of recall * 10000 (2 decimal places) */
	atomic_uint_fast64_t	validation_count;    /* Number of validations done */
	atomic_uint_fast64_t	total_used_k;        /* Sum of K used across all queries */
	atomic_uint_fast64_t	k_samples;           /* Number of K samples for avg */
}	t_lsh_stats_atomic;

/* DEPRECATED: Old non-atomic version - kept for compatibility */
typedef struct s_lsh_stats
{
	uint64_t	total_queries;       /* Total sparse attention queries */
	uint64_t	validated_queries;   /* Queries with brute-force validation */
	uint64_t	topk_hits;           /* Sum of (LSH ∩ BruteForce) tokens */
	uint64_t	topk_total;          /* Sum of K across validated queries */
	float		accumulated_recall;  /* Sum of recall values for averaging */
	uint64_t	validation_count;    /* Number of validations done */
	float		avg_recall;          /* Running average recall (0.0 - 1.0) */
	float		avg_dynamic_k;       /* Average K after adaptive cutoff */
	float		k_savings_pct;       /* Compute saved via adaptive K */
	uint64_t	total_used_k;        /* Sum of K used across all queries */
}	t_lsh_stats;

/* Thread-safe stats update macros */
# define LSH_STATS_INC_QUERY(s) atomic_fetch_add(&(s)->total_queries, 1)
# define LSH_STATS_ADD_USED_K(s, k) atomic_fetch_add(&(s)->total_used_k, (k))
# define LSH_STATS_ADD_K_SAMPLE(s) atomic_fetch_add(&(s)->k_samples, 1)
# define LSH_STATS_ADD_VALIDATION(s, hits, total, recall_pct) do { \
	atomic_fetch_add(&(s)->validated_queries, 1); \
	atomic_fetch_add(&(s)->topk_hits, (hits)); \
	atomic_fetch_add(&(s)->topk_total, (total)); \
	atomic_fetch_add(&(s)->recall_scaled, (uint64_t)((recall_pct) * 10000)); \
	atomic_fetch_add(&(s)->validation_count, 1); \
} while (0)

/* Thread-safe stats API (Phase 9) */
void		lsh_stats_reset_atomic(t_lsh_stats_atomic *s);
void		lsh_stats_print_atomic(const t_lsh_stats_atomic *s, int sparse_k);

/*
** Validate LSH candidates against brute-force (EXPENSIVE!)
** Call periodically (e.g., every 100 queries) for quality monitoring.
**
** @param query:     Query vector [dim]
** @param keys:      All key vectors [n_keys x dim] (BF16)
** @param n_keys:    Number of keys
** @param dim:       Vector dimension
** @param lsh_indices: Indices returned by LSH
** @param lsh_k:     Number of LSH candidates
** @param actual_k:  Top-K to validate against
** @return:          Recall (0.0 - 1.0)
*/
float		lsh_validate_recall(
				const float *query,
				const uint16_t *keys,
				int n_keys,
				int dim,
				const int *lsh_indices,
				int lsh_k,
				int actual_k);

/*
** Adaptive K: Select tokens until 95% probability mass covered
** Returns the number of tokens actually needed (dynamic_k <= max_k)
**
** @param scores:    Softmax-normalized attention scores [n_candidates]
** @param indices:   Candidate indices (sorted by score descending)
** @param n_cand:    Number of candidates
** @param threshold: Cumulative probability cutoff (e.g., 0.95)
** @return:          Number of tokens covering threshold probability
*/
int			adaptive_k_cutoff(
				const float *scores,
				const int *indices,
				int n_cand,
				float threshold);

#endif
