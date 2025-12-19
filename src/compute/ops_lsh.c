/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_lsh.c                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 12:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/16 00:00:00 by fgroo            ###   ########.fr       */
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
** Phase 10: Also creates transposed layout for SIMD turbo hash
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
	/* Phase 10 TURBO: Transpose to [dim][16] for column-major SIMD */
	d = 0;
	while (d < dim)
	{
		h = 0;
		while (h < LSH_NUM_HASHES)
		{
			ctx->hp_transposed[d][h] = ctx->hyperplanes[h][d];
			h++;
		}
		d++;
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
** Reset LSH index to empty state (call when clearing KV cache!)
** This prevents "zombie routing" where new tokens get routed to
** old, deleted key positions after a context reset.
*/
void	lsh_index_reset(t_lsh_index *idx)
{
	/* Just re-use the init logic - they're functionally identical */
	lsh_index_init(idx);
}

/*
** ============================================================================
** LSH HASH TURBO (Phase 10): Column-Major SIMD
** ============================================================================
** Old: 16 separate dot products (16 loops × dim iterations each)
** New: 1 loop × dim iterations with 16-wide FMA accumulator
**
** Strategy:
**   - Use 2×__m256 to hold 16 dot product accumulators (sums[0..7], sums[8..15])
**   - For each dimension d:
**       1. Broadcast v[d] to all 16 lanes (2× _mm256_set1_ps)
**       2. Load hp_transposed[d][0..15] (2× contiguous _mm256_loadu_ps)
**       3. FMA: sums += v[d]*hp[d]
**   - Final: Extract signs of 16 sums to form 16-bit hash
**
** Theoretical speedup: 16× fewer horizontal sums, better cache locality
** ============================================================================
*/
uint16_t	lsh_hash(const t_lsh_ctx *ctx, const float *vec)
{
	int			d;
	uint16_t	hash;

#ifdef __AVX512F__
	/* AVX-512: 16 floats in one register! */
	__m512		sums_512;
	__m512		v_broadcast;
	__m512		hp_vec;
	__mmask16	sign_mask;

	sums_512 = _mm512_setzero_ps();
	d = 0;
	while (d < ctx->dim)
	{
		v_broadcast = _mm512_set1_ps(vec[d]);
		hp_vec = _mm512_loadu_ps(ctx->hp_transposed[d]);
		sums_512 = _mm512_fmadd_ps(v_broadcast, hp_vec, sums_512);
		d++;
	}
	/* Extract sign bits: bit i = 1 if sums[i] > 0 */
	sign_mask = _mm512_cmp_ps_mask(sums_512, _mm512_setzero_ps(), _CMP_GT_OQ);
	hash = (uint16_t)sign_mask;

#elif defined(__AVX2__)
	/* AVX2: Use 2×__m256 for 16 accumulators */
	__m256	sums_lo;
	__m256	sums_hi;
	__m256	v_broadcast;
	__m256	hp_lo;
	__m256	hp_hi;
	__m256	zero;
	__m256	cmp_lo;
	__m256	cmp_hi;
	int		mask_lo;
	int		mask_hi;

	sums_lo = _mm256_setzero_ps();  /* Accumulators for hashes 0-7 */
	sums_hi = _mm256_setzero_ps();  /* Accumulators for hashes 8-15 */

	d = 0;
	while (d < ctx->dim)
	{
		/* Broadcast v[d] to all 8 lanes */
		v_broadcast = _mm256_set1_ps(vec[d]);
		/* Load hp_transposed[d][0..7] and [8..15] (contiguous!) */
		hp_lo = _mm256_loadu_ps(&ctx->hp_transposed[d][0]);
		hp_hi = _mm256_loadu_ps(&ctx->hp_transposed[d][8]);
		/* FMA: sums += v[d] * hp[d] */
		sums_lo = _mm256_fmadd_ps(v_broadcast, hp_lo, sums_lo);
		sums_hi = _mm256_fmadd_ps(v_broadcast, hp_hi, sums_hi);
		d++;
	}
	/* Extract sign bits: compare > 0, movemask */
	zero = _mm256_setzero_ps();
	cmp_lo = _mm256_cmp_ps(sums_lo, zero, _CMP_GT_OQ);
	cmp_hi = _mm256_cmp_ps(sums_hi, zero, _CMP_GT_OQ);
	mask_lo = _mm256_movemask_ps(cmp_lo);  /* Bits 0-7 */
	mask_hi = _mm256_movemask_ps(cmp_hi);  /* Bits 8-15 */
	hash = (uint16_t)(mask_lo | (mask_hi << 8));

#else
	/* Scalar fallback */
	float	sums[LSH_NUM_HASHES];
	int		h;

	h = 0;
	while (h < LSH_NUM_HASHES)
	{
		sums[h] = 0.0f;
		h++;
	}
	d = 0;
	while (d < ctx->dim)
	{
		float	v_d = vec[d];
		h = 0;
		while (h < LSH_NUM_HASHES)
		{
			sums[h] += v_d * ctx->hp_transposed[d][h];
			h++;
		}
		d++;
	}
	hash = 0;
	h = 0;
	while (h < LSH_NUM_HASHES)
	{
		if (sums[h] > 0.0f)
			hash |= (1U << h);
		h++;
	}
#endif

	return (hash);
}

/*
** Hamming distance (popcount of XOR)
** Uses hardware POPCNT instruction via __builtin_popcount (~1 cycle vs ~20)
*/
int	lsh_hamming_distance(uint16_t h1, uint16_t h2)
{
	return (__builtin_popcount((unsigned int)(h1 ^ h2)));
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
**
** CRITICAL FIX: Protected with omp critical to prevent race conditions.
** Multiple threads updating active block concurrently caused index corruption.
*/
void	lsh_update(t_lsh_index *idx, const t_lsh_ctx *ctx,
			const float *key, int dim, int pos)
{
	t_block_acc	*acc;
	float		centroid[LSH_MAX_DIM];
	float		scale;
	uint16_t	hash;
	int			d;

	/*
	** THREAD SAFETY: The active block accumulator is shared state.
	** Without synchronization, concurrent updates cause:
	** - Corrupted centroid sums (partial updates interleaved)
	** - Lost block boundaries (incorrect start_pos/count)
	** - Missed block commits (count check races)
	*/
	#pragma omp critical(lsh_update_lock)
	{
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
	}  /* end omp critical */
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

/* ===========================================================================
** LSH DIAGNOSTICS (Phase 9: Thread-Safe Stats)
** ===========================================================================
** CRITICAL FIX: Removed __thread which was losing stats when worker threads
** died. Stats now passed explicitly via t_lsh_stats_atomic pointer.
*/

#include <stdio.h>
#include "ops_simd.h"  /* bf16_to_f32_safe */
#include "../memory/safe_alloc.h"  /* xmalloc */

/*
** Reset atomic stats (called at start of benchmark/session)
** @param s: Pointer to atomic stats struct in transformer
*/
void	lsh_stats_reset_atomic(t_lsh_stats_atomic *s)
{
	if (!s)
		return ;
	atomic_store(&s->total_queries, 0);
	atomic_store(&s->validated_queries, 0);
	atomic_store(&s->topk_hits, 0);
	atomic_store(&s->topk_total, 0);
	atomic_store(&s->recall_scaled, 0);
	atomic_store(&s->validation_count, 0);
	atomic_store(&s->total_used_k, 0);
	atomic_store(&s->k_samples, 0);
}

/*
** Print atomic stats summary
** @param s: Pointer to atomic stats struct
** @param sparse_k: Configured sparse_k for savings calculation
*/
void	lsh_stats_print_atomic(const t_lsh_stats_atomic *s, int sparse_k)
{
	uint64_t	total_q;
	uint64_t	validated;
	uint64_t	hits;
	uint64_t	total;
	uint64_t	used_k;
	uint64_t	k_samples;
	float		recall;
	float		avg_k;
	float		savings;

	if (!s)
		return ;
	total_q = atomic_load(&s->total_queries);
	validated = atomic_load(&s->validated_queries);
	hits = atomic_load(&s->topk_hits);
	total = atomic_load(&s->topk_total);
	used_k = atomic_load(&s->total_used_k);
	k_samples = atomic_load(&s->k_samples);
	if (validated == 0 || total == 0)
	{
		printf("[LSH Stats] No validation data collected.\n");
		return ;
	}
	recall = (float)hits / (float)total;
	avg_k = (k_samples > 0) ? (float)used_k / (float)k_samples : 0.0f;
	savings = (sparse_k > 0) ? (1.0f - avg_k / (float)sparse_k) * 100.0f : 0.0f;
	printf("[LSH Stats] Queries: %lu (validated: %lu)\n",
		(unsigned long)total_q, (unsigned long)validated);
	printf("[LSH Stats] Recall: %.1f%% (target: >80%%)\n", recall * 100.0f);
	printf("[LSH Stats] Avg Dynamic K: %.1f (savings: %.1f%%)\n", avg_k, savings);
	if (recall < 0.8f)
		printf("[LSH Stats] ⚠️  WARNING: Recall below 80%% - sparse attention degraded!\n");
	else
		printf("[LSH Stats] ✅ Recall acceptable\n");
}

/*
** Brute-force compute true Top-K for validation (EXPENSIVE!)
** Used only periodically to monitor LSH quality
*/
static void	bf_compute_topk(
	const float *query,
	const uint16_t *keys,
	int n_keys,
	int dim,
	int k,
	int *topk_indices)
{
	float	*scores;
	int		i;
	int		j;
	int		d;
	float	dot;
	float	max_score;
	int		max_idx;

	scores = (float *)malloc(n_keys * sizeof(float));
	if (!scores)
		return ;

	/* Compute all dot products */
	i = 0;
	while (i < n_keys)
	{
		dot = 0.0f;
		d = 0;
		while (d < dim)
		{
			dot += query[d] * bf16_to_f32_safe(keys[i * dim + d]);
			d++;
		}
		scores[i] = dot;
		i++;
	}

	/* Find Top-K via repeated argmax (O(n*k), fine for validation) */
	j = 0;
	while (j < k)
	{
		max_score = -1e30f;
		max_idx = 0;
		i = 0;
		while (i < n_keys)
		{
			if (scores[i] > max_score)
			{
				max_score = scores[i];
				max_idx = i;
			}
			i++;
		}
		topk_indices[j] = max_idx;
		scores[max_idx] = -1e30f;  /* Mark as used */
		j++;
	}
	free(scores);
}

float	lsh_validate_recall(
	const float *query,
	const uint16_t *keys,
	int n_keys,
	int dim,
	const int *lsh_indices,
	int lsh_k,
	int actual_k)
{
	int		*bf_topk;
	int		hits;
	int		i;
	int		j;
	float	recall;

	if (actual_k > n_keys)
		actual_k = n_keys;
	if (lsh_k > actual_k)
		lsh_k = actual_k;

	bf_topk = (int *)malloc(actual_k * sizeof(int));
	if (!bf_topk)
		return (0.0f);

	/* Get brute-force Top-K */
	bf_compute_topk(query, keys, n_keys, dim, actual_k, bf_topk);

	/* Count hits: how many LSH indices are in BF Top-K? */
	hits = 0;
	i = 0;
	while (i < lsh_k)
	{
		j = 0;
		while (j < actual_k)
		{
			if (lsh_indices[i] == bf_topk[j])
			{
				hits++;
				break ;
			}
			j++;
		}
		i++;
	}
	free(bf_topk);

	recall = (float)hits / (float)actual_k;

	/*
	** NOTE: Stats update moved to caller via atomic macros.
	** Caller should use: LSH_STATS_ADD_VALIDATION(&t->lsh_stats, hits, actual_k, recall)
	*/

	return (recall);
}

/*
** Adaptive K: Find minimum K covering threshold probability mass
** Assumes scores are softmax-normalized and indices are sorted by score desc
*/
int	adaptive_k_cutoff(
	const float *scores,
	const int *indices,
	int n_cand,
	float threshold)
{
	float	cum_prob;
	int		k;

	(void)indices;  /* Indices already sorted, we just iterate scores */
	cum_prob = 0.0f;
	k = 0;
	while (k < n_cand)
	{
		cum_prob += scores[k];
		k++;
		if (cum_prob >= threshold)
			break ;
	}
	/* Minimum 1 token */
	if (k < 1)
		k = 1;
	return (k);
}

