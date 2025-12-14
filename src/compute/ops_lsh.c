/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_lsh.c                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 12:00:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 12:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_lsh.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/*
** Box-Muller transform for Gaussian random values
*/
static float	rand_gaussian(unsigned int *seed)
{
	float	u1;
	float	u2;

	*seed = *seed * 1103515245 + 12345;
	u1 = (float)((*seed >> 16) & 0x7FFF) / 32767.0f;
	*seed = *seed * 1103515245 + 12345;
	u2 = (float)((*seed >> 16) & 0x7FFF) / 32767.0f;
	if (u1 < 1e-6f)
		u1 = 1e-6f;
	return (sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2));
}

/*
** Initialize LSH context with random hyperplanes
*/
void	lsh_init(t_lsh_ctx *ctx, int dim, unsigned int seed)
{
	int		h;
	int		d;
	float	norm;

	if (dim > LSH_MAX_DIM)
		dim = LSH_MAX_DIM;
	ctx->dim = dim;
	h = 0;
	while (h < LSH_NUM_HASHES)
	{
		norm = 0.0f;
		d = 0;
		while (d < dim)
		{
			ctx->hyperplanes[h][d] = rand_gaussian(&seed);
			norm += ctx->hyperplanes[h][d] * ctx->hyperplanes[h][d];
			d++;
		}
		norm = sqrtf(norm);
		if (norm > 1e-6f)
		{
			d = 0;
			while (d < dim)
			{
				ctx->hyperplanes[h][d] /= norm;
				d++;
			}
		}
		h++;
	}
	ctx->initialized = 1;
}

/*
** Initialize empty index with all buckets empty
*/
void	lsh_index_init(t_lsh_index *idx)
{
	int	i;

	idx->n_blocks = 0;
	idx->block_size = LSH_BLOCK_SIZE;
	/* Mark all buckets empty */
	i = 0;
	while (i < LSH_NUM_BUCKETS)
	{
		idx->bucket_heads[i] = -1;
		i++;
	}
	/* Clear active block accumulator */
	memset(idx->active.sum_buffer, 0, sizeof(idx->active.sum_buffer));
	idx->active.count = 0;
	idx->active.start_pos = 0;
}

/*
** Compute LSH hash using SimHash (sign of dot products)
*/
uint16_t	lsh_hash(const t_lsh_ctx *ctx, const float *vec)
{
	uint16_t	hash;
	float		dot;
	int			h;
	int			d;

	hash = 0;
	h = 0;
	while (h < LSH_NUM_HASHES)
	{
		dot = 0.0f;
		/* SIMD dot product */
		__m256 sum_vec = _mm256_setzero_ps();
		d = 0;
		while (d + 7 < ctx->dim)
		{
			__m256 v = _mm256_loadu_ps(vec + d);
			__m256 hp = _mm256_loadu_ps(ctx->hyperplanes[h] + d);
			sum_vec = _mm256_fmadd_ps(v, hp, sum_vec);
			d += 8;
		}
		/* Horizontal sum */
		__m128 lo = _mm256_castps256_ps128(sum_vec);
		__m128 hi = _mm256_extractf128_ps(sum_vec, 1);
		lo = _mm_add_ps(lo, hi);
		lo = _mm_hadd_ps(lo, lo);
		lo = _mm_hadd_ps(lo, lo);
		dot = _mm_cvtss_f32(lo);
		/* Scalar remainder */
		while (d < ctx->dim)
		{
			dot += vec[d] * ctx->hyperplanes[h][d];
			d++;
		}
		if (dot > 0.0f)
			hash |= (1U << h);
		h++;
	}
	return (hash);
}

/*
** Hamming distance (popcount of XOR)
*/
int	lsh_hamming_distance(uint16_t h1, uint16_t h2)
{
	uint16_t	diff;
	int			count;

	diff = h1 ^ h2;
	count = 0;
	while (diff)
	{
		count += (diff & 1);
		diff >>= 1;
	}
	return (count);
}

/*
** Insert block into hash table
*/
void	lsh_insert_block(t_lsh_index *idx, uint16_t hash, int start_pos, int len)
{
	int	block_id;
	int	bucket;

	if (idx->n_blocks >= LSH_MAX_BLOCKS)
		return ;
	block_id = idx->n_blocks;
	idx->block_hashes[block_id] = hash;
	idx->block_starts[block_id] = start_pos;
	idx->block_lens[block_id] = len;
	/* Insert into hash table (linked list at bucket) */
	bucket = hash % LSH_NUM_BUCKETS;
	idx->chain_next[block_id] = idx->bucket_heads[bucket];
	idx->bucket_heads[bucket] = block_id;
	idx->n_blocks++;
}

/*
** Incremental update: called for each new key added to KV cache
** Accumulates keys, commits block when full
*/
void	lsh_update(t_lsh_index *idx, const t_lsh_ctx *ctx,
			const float *key, int dim, int pos)
{
	t_block_acc	*acc;
	float		centroid[LSH_MAX_DIM];
	float		scale;
	uint16_t	hash;
	int			d;

	acc = &idx->active;
	/* Track start position of new block */
	if (acc->count == 0)
		acc->start_pos = pos;
	/* Accumulate key into sum buffer (SIMD) */
	d = 0;
	while (d + 7 < dim && d < LSH_MAX_DIM)
	{
		__m256 sum = _mm256_loadu_ps(acc->sum_buffer + d);
		__m256 k = _mm256_loadu_ps(key + d);
		_mm256_storeu_ps(acc->sum_buffer + d, _mm256_add_ps(sum, k));
		d += 8;
	}
	while (d < dim && d < LSH_MAX_DIM)
	{
		acc->sum_buffer[d] += key[d];
		d++;
	}
	acc->count++;
	/* Block full? Commit to index! */
	if (acc->count >= LSH_BLOCK_SIZE)
	{
		/* Compute centroid = sum / count */
		scale = 1.0f / (float)acc->count;
		d = 0;
		while (d < dim && d < LSH_MAX_DIM)
		{
			centroid[d] = acc->sum_buffer[d] * scale;
			d++;
		}
		/* Hash centroid */
		hash = lsh_hash(ctx, centroid);
		/* Insert into index */
		lsh_insert_block(idx, hash, acc->start_pos, acc->count);
		/* Reset accumulator */
		memset(acc->sum_buffer, 0, sizeof(acc->sum_buffer));
		acc->count = 0;
	}
}

/*
** Find candidate blocks using BUCKET LOOKUP (not linear scan!)
** Multi-probe LSH: Check exact bucket + nearby buckets via bit flips
*/
int	lsh_find_candidates(const t_lsh_index *idx, uint16_t query_hash,
		int *block_ids, int max_blocks)
{
	int			n_found;
	int			b;
	int			block_id;
	int			probe;
	int			bit;
	uint16_t	probed_hash;
	int			bucket;

	n_found = 0;
	
	/* Multi-probe LSH: probe exact bucket + bit-flip neighbors */
	/* Probe 0 = exact match, Probes 1-16 = single bit flips */
	probe = 0;
	while (probe <= LSH_NUM_HASHES && n_found < max_blocks)
	{
		if (probe == 0)
			probed_hash = query_hash;
		else
		{
			/* Flip bit (probe-1) to probe nearby buckets */
			bit = probe - 1;
			probed_hash = query_hash ^ (1U << bit);
		}
		
		/* O(1) bucket lookup */
		bucket = probed_hash % LSH_NUM_BUCKETS;
		block_id = idx->bucket_heads[bucket];
		
		/* Traverse chain (average length << n_blocks) */
		while (block_id != -1 && n_found < max_blocks)
		{
			/* Verify Hamming distance (bucket collision possible) */
			if (lsh_hamming_distance(query_hash, idx->block_hashes[block_id])
				<= LSH_HAMMING_RADIUS)
			{
				/* Check for duplicates (multi-probe may find same block) */
				int dup = 0;
				for (b = 0; b < n_found && !dup; b++)
					if (block_ids[b] == block_id)
						dup = 1;
				if (!dup)
					block_ids[n_found++] = block_id;
			}
			block_id = idx->chain_next[block_id];
		}
		probe++;
	}
	return (n_found);
}

/*
** Get active (uncommitted) block info - MUST always be scanned
*/
int	lsh_get_active_block(const t_lsh_index *idx, int *start_pos, int *len)
{
	if (idx->active.count == 0)
		return (0);
	*start_pos = idx->active.start_pos;
	*len = idx->active.count;
	return (1);
}
