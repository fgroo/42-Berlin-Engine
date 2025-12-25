#include "inference.h"
#include "../config.h"
#include "compute/ops.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // AVX2 intrinsics
#include <omp.h>        // OpenMP parallelization
#include "compute/ops_lsh.h"  // LSH for sparse attention routing
#include "compute/ops_heap.h"  // Min-heap for O(n log k) Top-K selection
#include "compute/ops_attention.h"  // Flash Attention kernel
#include "compute/simd_kernels.h"  // SIMD primitives (simd_dot_bf16_f32, etc.)
#include "compute/ops_math_fast.h"  // fast_expf for SiLU activation
#include "compute/ops_rope.h"  // Phase 12: Precomputed RoPE sin/cos tables

/*
** ============================================================================
** CACHE-FRIENDLY CANDIDATE SORTING (Perf #6)
** ============================================================================
** Sorting candidate_positions before key access improves hardware prefetch
** hits. Memory access pattern goes from random to sequential, gaining 2-4x
** on cache-bound workloads.
*/
static int	int_compare_asc(const void *a, const void *b)
{
	return (*(const int *)a - *(const int *)b);
}

/* LSH stats now in transformer struct - use t->lsh_stats (Phase 9) */

// Helper: Pack 8x 32-bit ints to 8x 16-bit ints (lower 16 bits of each)
// AVX2 doesn't have direct 256->128 pack, so we extract and combine
static inline __m128i _mm256_cvtepi32_epi16_custom(__m256i a)
{
	// Extract lower and upper 128 bits
	__m128i lo = _mm256_castsi256_si128(a);
	__m128i hi = _mm256_extracti128_si256(a, 1);
	// Pack 32->16 with saturation (we use unsigned, values are small)
	return _mm_packus_epi32(lo, hi);
}

// Helper to wrap buffer in tensor
// Shape is [dim, 1] for column vector compatibility with matmul W[m,k] x x[k,1]
// NOTE: op_rmsnorm needs special handling since it expects feature dim as last axis
static void wrap_tensor(t_tensor *t, float *data, int dim)
{
	t->data = data;
	t->ndim = 2;
	t->shape[0] = dim;  // feature dimension
	t->shape[1] = 1;    // batch size = 1
	t->stride[0] = 1;
	t->stride[1] = 1;
	t->size = dim;  // CRITICAL: Set size for ops that use it
	t->dtype = DTYPE_F32; // Activations are F32
}



float	*transformer_forward(t_transformer *t, int token, int pos)
{
	t_transformer_config *c = &t->config;
	t_inference_state *s = &t->state;
	t_tensor x_tensor, xb_tensor, hb_tensor, hb2_tensor;
	t_tensor q_tensor, k_tensor, v_tensor;
	t_tensor logits_tensor;
	
	/* Record token in history for context-aware backward pass */
	if (s->token_history && pos >= 0 && pos < c->seq_len)
		s->token_history[pos] = token;

	/* DEBUG: Trace which model is being called (disabled for production)
	printf("[FWD] Entry: dim=%d, layers=%d, heads=%d, token=%d, pos=%d\n",
		c->dim, c->n_layers, c->n_heads, token, pos);
	*/

	// 1. Embedding
	t_tensor *emb = t->weights.token_embedding;
	if (!emb || !emb->data) {
		printf("[FWD] FATAL: No embedding data!\n");
		exit(1);
	}
	uint16_t *emb_data = (uint16_t *)emb->data;
	uint16_t *token_vec = emb_data + token * c->dim;
	/* printf("[FWD] Embedding OK (vocab offset=%d)\n", token * c->dim); */
	
	// [HOTFIX] Issue #5: Use SIMD for BF16→F32 conversion (was scalar)
	simd_bf16_to_f32(s->x, token_vec, c->dim);

	wrap_tensor(&x_tensor, s->x, c->dim);
	
	// DEBUG: RoPE / Pos check
	// printf("[DEBUG] Decoding Step: pos=%d, token=%d\n", pos, token);
	wrap_tensor(&xb_tensor, s->xb, c->dim);
	wrap_tensor(&hb_tensor, s->hb, c->hidden_dim);
	wrap_tensor(&hb2_tensor, s->hb2, c->hidden_dim);
	
	wrap_tensor(&q_tensor, s->q, c->n_heads * c->head_dim);
	wrap_tensor(&k_tensor, s->k, c->n_kv_heads * c->head_dim);
	wrap_tensor(&v_tensor, s->v, c->n_kv_heads * c->head_dim);
	
	// 2. Layers
	for (int i = 0; i < c->n_layers; i++)
	{
		// Progress indicator (disabled - floods output)
		// printf("."); fflush(stdout);

		// [PERF FIX] REMOVED REDUNDANT MEMSET
		// The attention kernels (op_attention_dense, op_attention_sparse, 
		// op_paged_attention) already zero their output buffers internally.
		// This was zeroing 16KB per layer = 416KB wasted memory traffic per token!
		// Verified by: bench_perf comparison before/after removal.
		
		t_layer_weights *l = &t->weights.layers[i];
		if (!l->wq || !l->wq->data || !l->wk || !l->wk->data || !l->wv || !l->wv->data || !l->wo || !l->wo->data) {
			printf("Missing QKV/O data L%d\n", i);
			exit(1);
		}
		if (!l->attention_norm || !l->attention_norm->data) {
			printf("Missing AttNorm L%d\n", i);
			exit(1);
		}
		if (!l->ffn_norm || !l->ffn_norm->data) {
			printf("Missing FFNNorm L%d\n", i);
			exit(1);
		}

		// RMSNorm
		op_rmsnorm(&xb_tensor, &x_tensor, l->attention_norm, c->norm_eps);
		


		// QKV Projections
		op_matmul(&q_tensor, l->wq, &xb_tensor);
		

		
		op_matmul(&k_tensor, l->wk, &xb_tensor);
		op_matmul(&v_tensor, l->wv, &xb_tensor);

		// RoPE: Apply rotary positional embedding
		// Phase 12: Use precomputed sin/cos tables if available (10-20% faster)
		if (t->rope_cache)
		{
			t_rope_cache *cache = (t_rope_cache *)t->rope_cache;
			// Single function call for all heads (much faster than per-head loop)
			rope_apply_multihead_cached(s->q, c->n_heads, pos, cache);
			rope_apply_multihead_cached(s->k, c->n_kv_heads, pos, cache);
		}
		else
		{
			// Fallback: Legacy path with per-token sinf/cosf (slower)
			t_rope_ctx rope_ctx = {
				.head_dim = c->head_dim,
				.theta_base = c->rope_theta,
				.beta_fast = c->beta_fast,
				.beta_slow = c->beta_slow,
				.factor = c->rope_factor,
				.mscale = (c->mscale == 1.0f && c->rope_factor > 1.0f) 
						  ? sqrtf(0.1f * logf(c->rope_factor) + 1.0f) 
						  : c->mscale,
				.orig_ctx = c->orig_ctx_len,
				.thetas_cache = t->rope_thetas
			};
			op_rope(&q_tensor, pos, &rope_ctx);
			op_rope(&k_tensor, pos, &rope_ctx);
		}

		// KV Cache Append: Use paged or linear based on config
		if (t->use_paged_kv)
			paged_kv_append(&t->paged_kv[i], s->k, s->v, pos);
		else
			kv_cache_append(&s->kv_cache[i], &k_tensor, &v_tensor);
		
		// ========== LSH INDEX UPDATE ==========
		// Incrementally build block index for true O(K) sparse attention
		// We accumulate keys, compute centroid + hash per block
		if (t->lsh_ctx && t->lsh_index && i == 0)  // Layer 0 only for routing
		{
			t_lsh_ctx *lsh_ctx = (t_lsh_ctx *)t->lsh_ctx;
			t_lsh_index *lsh_idx = (t_lsh_index *)t->lsh_index;
			/* [FIX #1] VLA REMOVAL: Use scratch arena instead of 32KB stack array
			** Old: float k_f32[LSH_MAX_DIM]; // 8192 floats = 32KB STACK BOMB!
			** This caused stack overflow on deep recursion / high dim models. */
			size_t lsh_scratch_saved = t->scratch.offset;
			float *k_f32 = arena_try_alloc(&t->scratch, c->head_dim * sizeof(float));
			if (k_f32)
			{
				for (int d = 0; d < c->head_dim; d++)
					k_f32[d] = s->k[d];
				lsh_update(lsh_idx, lsh_ctx, k_f32, c->head_dim, pos);
				t->scratch.offset = lsh_scratch_saved;
			}
			/* On OOM: skip LSH update (graceful degradation) */
		}
		
		// ========== KV CACHE EVICTION ==========
		// Auto-evict low-importance keys when cache fills up
		/* [FIX] Skip eviction if weights not allocated (OOM during init) */
		if (s->kv_cache[i].current_seq_len > t->evict_threshold && t->evict_weights.data)
		{
			t_evict_params ep = {
				.q = &q_tensor,
				.w = &t->evict_weights,
				.keep_k = t->evict_keep_k,
				.scratch = &t->scratch
			};
			kv_cache_evict(&s->kv_cache[i], &ep);
		}

		// ========== SPARSE ATTENTION ==========
		// Instead of attending to ALL previous tokens O(N²),
		// we use the Lightning Indexer to score keys cheaply,
		// then select Top-K most relevant keys.
		// This gives O(N*K) complexity for infinite context!
		
		// NOTE: Attention output buffer already zeroed at layer start (line 104)
		// to handle buffer aliasing between attention and FFN
		
		int kv_len = s->kv_cache[i].current_seq_len;
		int use_sparse = (t->sparse_k > 0 && kv_len > t->sparse_k);
		int attend_k = use_sparse ? t->sparse_k : kv_len;
		
		// Allocate indices array (on scratch arena for sparse, on stack for dense)
		int *topk_indices = NULL;
		size_t scratch_saved = t->scratch.offset;
		
		if (use_sparse)
		{
			// ========== HYBRID ATTENTION ==========
			// 1. Sliding Window (Local): Always attend to last W tokens
			//    - Guarantees local reasoning (arithmetic, syntax)
			// 2. LSH (Global): For older tokens, use hash-based retrieval
			//    - Handles long-context memory (RAG-style)
			
			/*
			** GRACEFUL OOM HANDLING:
			** If any arena allocation fails, we fall back to dense attention.
			** This prevents crashes on ultra-long contexts where arena is exhausted.
			** Better slow than dead.
			*/
			topk_indices = arena_try_alloc(&t->scratch, attend_k * sizeof(int));
			if (!topk_indices)
			{
				fprintf(stderr, "[WARN] Arena OOM for topk_indices. Fallback to dense.\n");
				use_sparse = 0;
				attend_k = kv_len;
				goto do_attention;  // Skip sparse logic
			}
			
			t_lsh_ctx *lsh_ctx = (t_lsh_ctx *)t->lsh_ctx;
			t_lsh_index *lsh_idx = (t_lsh_index *)t->lsh_index;
			int head_dim = c->head_dim;
			
			// Allocate candidate positions (window + LSH candidates)
			int max_candidates = kv_len;  // Worst case: all positions
			int *candidate_positions = arena_try_alloc(&t->scratch, max_candidates * sizeof(int));
			if (!candidate_positions)
			{
				fprintf(stderr, "[WARN] Arena OOM for candidates. Fallback to dense.\n");
				t->scratch.offset = scratch_saved;
				use_sparse = 0;
				attend_k = kv_len;
				goto do_attention;
			}
			int n_candidates = 0;
			
			// ===== STEP 1: SLIDING WINDOW (mandatory local context) =====
			// Always include the last ATTN_WINDOW_SIZE positions
			int window_start = (kv_len > ATTN_WINDOW_SIZE) ? (kv_len - ATTN_WINDOW_SIZE) : 0;
			for (int ti = window_start; ti < kv_len; ti++)
			{
				candidate_positions[n_candidates++] = ti;
			}
			
			// ===== STEP 2: LSH CANDIDATES (global context for old positions) =====
			// Only add LSH candidates for positions OLDER than the window
			if (window_start > 0 && lsh_ctx && lsh_idx && lsh_idx->n_blocks > 0)
			{
				/* [FIX #1] VLA REMOVAL: Use scratch arena instead of 32KB stack array
				** Old: float q_mean[LSH_MAX_DIM]; // 32KB STACK BOMB!
				** Now uses already-saved scratch with fallback to window-only attention. */
				float *q_mean = arena_try_alloc(&t->scratch, head_dim * sizeof(float));
				if (!q_mean)
				{
					/* Fallback: window-only attention (no LSH candidates) */
					fprintf(stderr, "[LSH] OOM for q_mean, using window-only\n");
					goto skip_lsh_candidates;
				}
				memset(q_mean, 0, head_dim * sizeof(float));
				for (int h = 0; h < c->n_heads; h++)
					for (int d = 0; d < head_dim; d++)
						q_mean[d] += s->q[h * head_dim + d];
				for (int d = 0; d < head_dim; d++)
					q_mean[d] /= c->n_heads;
				
				uint32_t q_hash = lsh_ctx->initialized ? lsh_hash(lsh_ctx, q_mean) : 0;
				
				// Find candidate blocks via LSH
				int block_ids[64];
				int n_candidate_blocks = lsh_find_candidates(lsh_idx, (uint16_t)q_hash, 
					block_ids, 64);
				
				// Add positions from LSH blocks (only if BEFORE window)
				for (int b = 0; b < n_candidate_blocks && n_candidates < max_candidates; b++)
				{
					int blk_start = lsh_idx->block_starts[block_ids[b]];
					int blk_len = lsh_idx->block_lens[block_ids[b]];
					// Only include if block ends before window starts
					if (blk_start + blk_len <= window_start)
					{
						for (int p = 0; p < blk_len && n_candidates < max_candidates; p++)
						{
							int pos_idx = blk_start + p;
							if (pos_idx < window_start)  // Don't duplicate window
								candidate_positions[n_candidates++] = pos_idx;
						}
					}
				}
			}
			
			skip_lsh_candidates:  /* [FIX #1] Fallback label for OOM in q_mean allocation */
			/* ===== STEP 2.5: Sort candidates for cache-friendly access ===== */
			/* Sequential memory access enables hardware prefetcher (2-4x speedup) */
			if (n_candidates > 1)
				qsort(candidate_positions, n_candidates, sizeof(int), int_compare_asc);
			
			// Step 3: Score only candidate positions with SIMD
			float *key_scores = arena_try_alloc(&t->scratch, n_candidates * sizeof(float));
			if (!key_scores)
			{
				fprintf(stderr, "[WARN] Arena OOM for key_scores. Fallback to dense.\n");
				t->scratch.offset = scratch_saved;
				use_sparse = 0;
				attend_k = kv_len;
				goto do_attention;
			}
			t_bf16 *k_data = (t_bf16 *)s->kv_cache[i].k.data;
			int kv_stride = c->n_kv_heads * c->head_dim;
			
			#pragma omp parallel for schedule(static)
			for (int ci = 0; ci < n_candidates; ci++)
			{
				int ti = candidate_positions[ci];
				float score = 0.0f;
				for (int kv_h = 0; kv_h < c->n_kv_heads; kv_h++)
				{
					t_bf16 *k_vec = k_data + ti * kv_stride + kv_h * head_dim;
					float *q_for_kv = s->q + (kv_h * (c->n_heads/c->n_kv_heads)) * head_dim;
					score += simd_dot_bf16_f32(k_vec, q_for_kv, head_dim);
				}
				key_scores[ci] = score;
			}
			
			/* ====== PHASE 9: LSH DIAGNOSTICS (Thread-Safe) ====== */
		/* Track stats atomically - no more __thread data loss! */
		LSH_STATS_INC_QUERY(&t->lsh_stats);

			#if DEBUG_LSH
			/* Validate recall periodically */
			{
				uint64_t query_count = atomic_load(&t->lsh_stats.total_queries);
				if (query_count % LSH_VALIDATION_INTERVAL == 0 && n_candidates > 0)
				{
					/* Get K vectors from first KV head for validation */
					t_bf16 *k_data_val = (t_bf16 *)s->kv_cache[i].k.data;
					int k_stride_val = c->n_kv_heads * head_dim;
					(void)k_stride_val;
					
					/* Call brute-force validation */
					float recall = lsh_validate_recall(
						s->q,                    /* Query vector (F32) */
						(const uint16_t *)k_data_val, /* All keys (BF16) */
						kv_len,                  /* Number of keys */
						head_dim,                /* Vector dimension */
						candidate_positions,     /* LSH candidate indices */
						n_candidates,            /* Number of candidates */
						(n_candidates < 32) ? n_candidates : 32  /* Top-32 */
					);
					
					/* Atomic stats update */
					int val_k = (n_candidates < 32) ? n_candidates : 32;
					LSH_STATS_ADD_VALIDATION(&t->lsh_stats, 
						(int)(recall * val_k), val_k, recall);
					
					/* Warn on low recall */
					if (recall < 0.70f && query_count % 1000 == 0)
					{
						fprintf(stderr, "[LSH WARN] Low recall: %.1f%% at pos %d\n",
							recall * 100.0f, pos);
					}
				}
			}
			#endif
			
			// Step 4: Select Top-K from candidates using MIN-HEAP
			// Complexity: O(n log k) vs O(n*k) for partial sort
			int actual_k = (n_candidates < attend_k) ? n_candidates : attend_k;
			
			// Allocate heap on scratch arena
			t_heap_item *heap = arena_try_alloc(&t->scratch, actual_k * sizeof(t_heap_item));
			if (!heap)
			{
				fprintf(stderr, "[WARN] Arena OOM for heap. Fallback to dense.\n");
				t->scratch.offset = scratch_saved;
				use_sparse = 0;
				attend_k = kv_len;
				goto do_attention;
			}
			int heap_size = 0;
			
			// Build heap with O(n log k) insertion
			for (int ci = 0; ci < n_candidates; ci++)
				heap_push(heap, &heap_size, actual_k, key_scores[ci], candidate_positions[ci]);
			
			// Extract indices from heap
			heap_extract_indices(heap, heap_size, topk_indices);
			attend_k = heap_size;
			
			/* ====== PHASE 9: Atomic K Tracking ====== */
			LSH_STATS_ADD_USED_K(&t->lsh_stats, attend_k);
			LSH_STATS_ADD_K_SAMPLE(&t->lsh_stats);
		}
		
		/*
		** do_attention: Label for OOM fallback from sparse attention path.
		** SAFETY: All gotos jumping here come from within the if(use_sparse) block.
		** Variables declared after those gotos (lsh_ctx, lsh_idx, head_dim, etc.)
		** are only used within the sparse block and are not accessed here.
		** This is a safe error-handling pattern - better slow (dense) than dead (OOM).
		*/
		do_attention:
		// ========== FLASH ATTENTION ==========
		// O(head_dim) memory instead of O(seq_len) for scores
		// Uses online softmax for numerical stability
		if (t->use_paged_kv)
		{
			// PAGED PATH: Block-based iteration
			float scale = 1.0f / sqrtf((float)c->head_dim);
			int n_blocks = t->paged_kv[i].n_blocks;
			
			// Sparse path: If we have more blocks than SPARSE_BLOCKS_K, select top-K
			#ifndef SPARSE_BLOCKS_K
			#define SPARSE_BLOCKS_K 8
			#endif
			
			if (n_blocks > SPARSE_BLOCKS_K)
			{
				// SPARSE PAGED PATH: O(K) attention instead of O(N)
				// Arena-allocate scores (safe for infinite context)
				size_t paged_scratch = t->scratch.offset;
				t_block_score *scores = arena_try_alloc(&t->scratch, 
					n_blocks * sizeof(t_block_score));
				(void)paged_scratch;  // Suppress unused warning
				if (!scores)
				{
					// Fallback to dense path if arena exhausted
					op_paged_attention(s->hb, s->q, &t->paged_kv[i],
						c->n_heads, c->n_kv_heads, c->head_dim, scale);
					continue ;
				}
				int selected[SPARSE_BLOCKS_K + 1];
				int n_selected;
				
				// Score blocks using Q·centroid (first query head)
				score_blocks(scores, s->q, &t->paged_kv[i], n_blocks);
				
				// Select top-K blocks (always includes last block for local window)
				n_selected = select_top_k_blocks(selected, scores, n_blocks, SPARSE_BLOCKS_K);

				/* ====== PHASE 9: PAGED SPARSE TRACKING (Atomic) ====== */
				#if DEBUG_LSH
				{
					LSH_STATS_INC_QUERY(&t->lsh_stats);
					LSH_STATS_ADD_USED_K(&t->lsh_stats, n_selected);
					LSH_STATS_ADD_K_SAMPLE(&t->lsh_stats);
					
					/* Paged path is exact - no LSH approximation */
					/* Recall is 100% since we score ALL blocks */
					LSH_STATS_ADD_VALIDATION(&t->lsh_stats, n_selected, n_selected, 1.0f);
				}
				#endif
				
				// Attend only to selected blocks
				op_paged_attention_sparse(s->hb, s->q, &t->block_manager,
					selected, n_selected, c->n_heads, c->n_kv_heads, c->head_dim, scale);
				
				// Restore arena
				t->scratch.offset = paged_scratch;
			}
			else
			{
				// DENSE PAGED PATH: Attend to all blocks
				op_paged_attention(s->hb, s->q, &t->paged_kv[i],
					c->n_heads, c->n_kv_heads, c->head_dim, scale);
			}
		}
		else
		{
			// LINEAR PATH: Original flat buffer
			t_attention_params attn = {
				.q = s->q,
				.kv_cache = &s->kv_cache[i],
				.output = s->hb,
				.n_heads = c->n_heads,
				.n_kv_heads = c->n_kv_heads,
				.head_dim = c->head_dim,
				.scale = 1.0f / sqrtf((float)c->head_dim),
				.topk_idx = topk_indices,
				.attend_k = attend_k
			};
			op_multihead_attention(&attn);
		}
		
		// Restore scratch arena
		if (use_sparse)
			t->scratch.offset = scratch_saved;
		
		// Output Projection
		t_tensor att_out_view;
		wrap_tensor(&att_out_view, s->hb, c->n_heads * c->head_dim);
		op_matmul(&xb_tensor, l->wo, &att_out_view);
		
		// Residual: x += xb (SIMD-accelerated)
		simd_add_f32(s->x, s->xb, c->dim);
		
		// FFN
		op_rmsnorm(&xb_tensor, &x_tensor, l->ffn_norm, c->norm_eps);
		
		
		
		// Gate & Up
		op_matmul(&hb_tensor, l->w1, &xb_tensor); // Gate
		op_matmul(&hb2_tensor, l->w3, &xb_tensor); // Up
		
		
		// SiLU * Up
		// op_silu_mul(out, gate, val)
		op_silu_mul(&hb_tensor, &hb_tensor, &hb2_tensor); // hb = silu(hb) * hb2
		
		
		// Down
		op_matmul(&xb_tensor, l->w2, &hb_tensor);
		
		
		// ========== NESTED LEARNING ADAPTER ==========
		// Apply fluid w2 adapter: xb += w2_adapter @ hb
		// This is where test-time learning injects context-specific knowledge
		// CRITICAL: Apply adapters ALWAYS (when they exist)
		// Only cache hb for backward pass when actively learning
		if (t->fluid_layers)
		{
			// Cache hb for backward pass ONLY during learning
			// SAFETY: Check hb_cache != NULL (frozen layers have NULL pointers!)
			if (t->nested_learning && t->fluid_layers[i].hb_cache)
				memcpy(t->fluid_layers[i].hb_cache, s->hb, c->hidden_dim * sizeof(float));
			
			// SIMD+OpenMP optimized adapter matmul: xb += SCALE * adapter @ hb
			// adapter is [dim x hidden_dim] (BF16), hb is [hidden_dim] (F32)
			// ADAPTER_SCALE amplifies the adapter's contribution (Solution 1)
			// NOTE: Disabled in favor of single final adapter (Solution 3+)
			/*
			t_tensor *adapter = t->fluid_layers[i].w2_weight;
			t_bf16 *a_data = (t_bf16 *)adapter->data;
			int hidden = c->hidden_dim;
			
			#pragma omp parallel for schedule(static)
			for (int row = 0; row < c->dim; row++)
			{
				t_bf16 *a_row = a_data + row * hidden;
				s->xb[row] += ADAPTER_SCALE * simd_dot_bf16_f32(a_row, s->hb, hidden);
			}
			*/
		}
		
		// Residual (SIMD-accelerated)
		simd_add_f32(s->x, s->xb, c->dim);
		
	}

	// 3. Final Norm
	op_rmsnorm(&x_tensor, &x_tensor, t->weights.norm, c->norm_eps);
	
	// ========== SOLUTION 4: FINAL HIDDEN ADAPTER [dim x dim] ==========
	// Apply learned adapter directly to final hidden state x
	// This uses properly-sized adapter with correct gradient flow
	if (t->final_adapter && t->final_adapter->data)
	{
		t_bf16 *adapter_data = (t_bf16 *)t->final_adapter->data;
		int dim = c->dim;
		
		// [HOTFIX] Issue #1: Use heap-allocated state field instead of static
		if (t->nested_learning)
			memcpy(s->final_input_cache, s->x, dim * sizeof(float));
		
		// [FIX] Use arena allocator instead of alloca() for stack safety
		// Scratch arena is reset per inference call, so this is safe
		size_t adapter_scratch_saved = t->scratch.offset;
		float *adapter_out = arena_try_alloc(&t->scratch, dim * sizeof(float));
		if (!adapter_out)
		{
			// Fallback: Apply adapter per-element without buffer
			// Slower but avoids fatal error on OOM
			fprintf(stderr, "[WARN] OOM for adapter_out buffer, using fallback\n");
			#pragma omp parallel for schedule(static)
			for (int row = 0; row < dim; row++)
			{
				t_bf16 *a_row = adapter_data + row * dim;
				float val = simd_dot_bf16_f32(a_row, s->x, dim);
				s->x[row] += ADAPTER_SCALE * val;
			}
		}
		else
		{
			#pragma omp parallel for schedule(static)
			for (int row = 0; row < dim; row++)
			{
				t_bf16 *a_row = adapter_data + row * dim;
				adapter_out[row] = simd_dot_bf16_f32(a_row, s->x, dim);
			}
			
			// [HOTFIX] Issue #2: Debug output now behind LOG_HOT macro
			#ifdef DEBUG_MODE
			{
				float max_contribution = 0.0f;
				for (int d = 0; d < dim; d++)
					if (fabsf(adapter_out[d]) > max_contribution)
						max_contribution = fabsf(adapter_out[d]);
				LOG_DEBUG("[FINAL_ADAPTER] max_out=%.6f, scaled=%.2f\n", 
					max_contribution, ADAPTER_SCALE * max_contribution);
			}
			#endif
			
			// [PHASE 12] PURE ADDITIVE INJECTION (LoRA-style)
			// x' = x + adapter * SCALE
			// NO base damping, NO sigmoid. With zero-init adapter, initial contribution = 0
			// Gradient grows adapter slowly from stable base model solution
			for (int d = 0; d < dim; d++)
				s->x[d] += ADAPTER_SCALE * adapter_out[d];
			
			// Restore scratch arena
			t->scratch.offset = adapter_scratch_saved;
		}
	}

	// 4. Logits
	wrap_tensor(&logits_tensor, s->logits, c->vocab_size);
	op_matmul(&logits_tensor, t->weights.output, &x_tensor);
	
	// Solution 5: Add logit bias directly to logits
	if (t->logit_bias)
	{
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < c->vocab_size; i++)
			s->logits[i] += t->logit_bias[i];
	}

	/* [PHASE 16] Context-Aware Bias Cache Lookup (Fixed Key Format) */
	/* Key = prev_token (unigram context), Target = what we predict */
	if (t->context_bias.keys)
	{
		/* We need the PREVIOUS token to look up what we should boost NOW */
		int prev_token = (pos > 0) ? t->state.token_history[pos - 1] : 1;
		uint64_t key = (uint64_t)prev_token;  // Unigram key matches training!
		
		// Simple hash: MurmurHash3-style mixer
		uint64_t h = key;
		h ^= h >> 33;
		h *= 0xff51afd7ed558ccdULL;
		h ^= h >> 33;
		h *= 0xc4ceb9fe1a85ec53ULL;
		h ^= h >> 33;
		uint32_t idx = (uint32_t)(h % t->context_bias.size);
		
		/* Linear probing */
		for (int i = 0; i < 16; i++) // Limit probing for speed
		{
			uint32_t cur = (idx + i) % t->context_bias.size;
			if (t->context_bias.keys[cur] == key)
			{
				/* [PHASE 16] Apply learned bias */
				float bias_val = t->context_bias.biases[cur];
				int target_tok = t->context_bias.tokens[cur];
				if (bias_val > 1.0f)
					printf("[CONTEXT_BIAS] HIT: After '%d' boost token %d by +%.2f\n",
						prev_token, target_tok, bias_val);
				s->logits[target_tok] += bias_val;
				break ;
			}
			if (t->context_bias.keys[cur] == 0)
				break ;
		}
	}

	return (s->logits);
}

/* NOTE: transformer_backward_step has been moved to src/nested/backward.c
** with FP32 gradient accumulation for mixed precision training.
** See backward_zero_grads() and backward_apply_grads() for new API.
*/

/*
** ============================================================================
** BATCHED PREFILL (Phase 2.2a)
** ============================================================================
** Process multiple tokens using GEMM instead of GEMV for projections.
** 
** Key insight: QKV projections are the same for all tokens in a batch.
** Instead of: for each token: Q[1,dim] = xb[1,dim] @ Wq[dim,dim]^T  (GEMV)
** We do:      Q[B,dim] = xb[B,dim] @ Wq[dim,dim]^T                   (GEMM)
**
** Attention is still sequential (causal mask requires it), but the
** expensive projections are now batched.
** ============================================================================
*/

void	forward_prefill_batch(t_transformer *t, const int *tokens,
		int batch_size, int start_pos)
{
	t_transformer_config	*c;
	t_inference_state		*s;
	int						b;
	int						i;
	int						dim;
	int						kv_dim;

	c = &t->config;
	s = &t->state;
	dim = c->dim;
	kv_dim = c->n_kv_heads * c->head_dim;

	/* Clamp batch size */
	if (batch_size <= 0)
		return ;
	if (batch_size > MAX_PREFILL_BATCH)
		batch_size = MAX_PREFILL_BATCH;

	/*
	** PHASE 1: Batched Embedding Lookup
	** Load all token embeddings into batch_x [batch_size x dim]
	*/
	#pragma omp parallel for schedule(static)
	for (b = 0; b < batch_size; b++)
	{
		int			token;
		int			j;
		uint16_t	*emb_data;
		uint16_t	*tok_vec;
		float		*dst;

		token = tokens[b];
		emb_data = (uint16_t *)t->weights.token_embedding->data;
		tok_vec = emb_data + token * dim;
		dst = s->batch_x + b * dim;

		/* BF16 -> F32 conversion */
		for (j = 0; j < dim; j++)
		{
			uint32_t val = (uint32_t)tok_vec[j] << 16;
			memcpy(&dst[j], &val, sizeof(float));
		}
	}

	/*
	** PHASE 2: Process each layer
	** Projections use GEMM, attention stays sequential
	*/
	for (i = 0; i < c->n_layers; i++)
	{
		t_layer_weights	*l;
		t_tensor		batch_xb_t;
		t_tensor		batch_q_t;
		t_tensor		batch_k_t;
		t_tensor		batch_v_t;

		l = &t->weights.layers[i];

		/*
		** 2a. Batched RMSNorm (AVX2 SIMD - Phase 9)
		** Now uses vectorized op_rmsnorm instead of scalar loop
		*/
		{
			t_tensor batch_x_t, batch_xb_out;
			
			/* Setup input tensor [dim x batch_size] */
			batch_x_t.data = s->batch_x;
			batch_x_t.ndim = 2;
			batch_x_t.shape[0] = dim;
			batch_x_t.shape[1] = batch_size;
			batch_x_t.size = dim * batch_size;
			batch_x_t.dtype = DTYPE_F32;
			
			/* Setup output tensor */
			batch_xb_out.data = s->batch_xb;
			batch_xb_out.ndim = 2;
			batch_xb_out.shape[0] = dim;
			batch_xb_out.shape[1] = batch_size;
			batch_xb_out.size = dim * batch_size;
			batch_xb_out.dtype = DTYPE_F32;
			
			/* AVX2 RMSNorm - processes all tokens in batch */
			op_rmsnorm(&batch_xb_out, &batch_x_t, l->attention_norm, c->norm_eps);
		}

		/*
		** 2b. Batched QKV Projections (GEMM!)
		** This is where we win: M=batch_size instead of M=1
		**
		** Q[batch_size, n_heads*head_dim] = batch_xb[batch_size, dim] @ Wq[dim, n_heads*head_dim]^T
		*/
		batch_xb_t.data = s->batch_xb;
		batch_xb_t.ndim = 2;
		batch_xb_t.shape[0] = dim;
		batch_xb_t.shape[1] = batch_size;  /* Column-major: dim rows, batch cols */
		batch_xb_t.size = dim * batch_size;
		batch_xb_t.dtype = DTYPE_F32;

		batch_q_t.data = s->batch_q;
		batch_q_t.ndim = 2;
		batch_q_t.shape[0] = c->n_heads * c->head_dim;
		batch_q_t.shape[1] = batch_size;
		batch_q_t.size = c->n_heads * c->head_dim * batch_size;
		batch_q_t.dtype = DTYPE_F32;

		batch_k_t.data = s->batch_k;
		batch_k_t.ndim = 2;
		batch_k_t.shape[0] = kv_dim;
		batch_k_t.shape[1] = batch_size;
		batch_k_t.size = kv_dim * batch_size;
		batch_k_t.dtype = DTYPE_F32;

		batch_v_t.data = s->batch_v;
		batch_v_t.ndim = 2;
		batch_v_t.shape[0] = kv_dim;
		batch_v_t.shape[1] = batch_size;
		batch_v_t.size = kv_dim * batch_size;
		batch_v_t.dtype = DTYPE_F32;

		/* GEMM: uses gemm_bf16_f32_tiled for M>1 */
		op_matmul(&batch_q_t, l->wq, &batch_xb_t);
		op_matmul(&batch_k_t, l->wk, &batch_xb_t);
		op_matmul(&batch_v_t, l->wv, &batch_xb_t);

		/*
		** 2c. RoPE + KV Cache + Attention (Sequential)
		** Attention is causal: token b can only attend to [0, start_pos+b]
		** So we process tokens one by one here
		*/
		for (b = 0; b < batch_size; b++)
		{
			int		pos;
			float	*q_row;
			float	*k_row;
			float	*v_row;

			pos = start_pos + b;
			q_row = s->batch_q + b * c->n_heads * c->head_dim;
			k_row = s->batch_k + b * kv_dim;
			v_row = s->batch_v + b * kv_dim;

			/* Copy to single-token buffers for existing ops */
			memcpy(s->q, q_row, c->n_heads * c->head_dim * sizeof(float));
			memcpy(s->k, k_row, kv_dim * sizeof(float));
			memcpy(s->v, v_row, kv_dim * sizeof(float));

			/* RoPE - Phase 12: Use precomputed cache if available */
			if (t->rope_cache)
			{
				t_rope_cache *cache = (t_rope_cache *)t->rope_cache;
				rope_apply_multihead_cached(s->q, c->n_heads, pos, cache);
				rope_apply_multihead_cached(s->k, c->n_kv_heads, pos, cache);
			}
			else
			{
				t_tensor q_t, k_t;
				q_t.data = s->q;
				q_t.size = c->n_heads * c->head_dim;
				q_t.dtype = DTYPE_F32;
				k_t.data = s->k;
				k_t.size = kv_dim;
				k_t.dtype = DTYPE_F32;

				t_rope_ctx rope_ctx = {
					.head_dim = c->head_dim,
					.theta_base = c->rope_theta,
					.beta_fast = c->beta_fast,
					.beta_slow = c->beta_slow,
					.factor = c->rope_factor,
					.mscale = (c->mscale == 1.0f && c->rope_factor > 1.0f)
						? sqrtf(0.1f * logf(c->rope_factor) + 1.0f)
						: c->mscale,
					.orig_ctx = c->orig_ctx_len,
					.thetas_cache = t->rope_thetas
				};
				op_rope(&q_t, pos, &rope_ctx);
				op_rope(&k_t, pos, &rope_ctx);
			}

			/* KV Cache Append */
			t_tensor k_tensor, v_tensor;
			k_tensor.data = s->k;
			k_tensor.size = kv_dim;
			k_tensor.dtype = DTYPE_F32;
			v_tensor.data = s->v;
			v_tensor.size = kv_dim;
			v_tensor.dtype = DTYPE_F32;

			if (t->use_paged_kv)
				paged_kv_append(&t->paged_kv[i], s->k, s->v, pos);
			else
				kv_cache_append(&s->kv_cache[i], &k_tensor, &v_tensor);

			/* Attention (uses existing dense/sparse path) */
			memset(s->hb, 0, c->n_heads * c->head_dim * sizeof(float));

			int kv_len = pos + 1;
			t_attention_params attn_p = {
				.output = s->hb,
				.q = s->q,
				.kv_cache = &s->kv_cache[i],
				.n_heads = c->n_heads,
				.n_kv_heads = c->n_kv_heads,
				.head_dim = c->head_dim,
				.scale = 1.0f / sqrtf((float)c->head_dim)
			};
			op_attention_dense(&attn_p, kv_len);

			/* Output projection: s->xb = Wo @ s->hb */
			/* Note: hb after attention is [n_heads * head_dim], not [dim] */
			int attn_dim = c->n_heads * c->head_dim;
			t_tensor hb_t, xb_t;
			hb_t.data = s->hb;
			hb_t.ndim = 2;
			hb_t.shape[0] = attn_dim;  /* Attention output = n_heads*head_dim = 4096 */
			hb_t.shape[1] = 1;
			hb_t.size = attn_dim;
			hb_t.dtype = DTYPE_F32;
			xb_t.data = s->xb;
			xb_t.ndim = 2;
			xb_t.shape[0] = dim;
			xb_t.shape[1] = 1;
			xb_t.size = dim;
			xb_t.dtype = DTYPE_F32;

				op_matmul(&xb_t, l->wo, &hb_t);

			/* Residual */
			float *x_row = s->batch_x + b * dim;
			for (int j = 0; j < dim; j++)
				x_row[j] += s->xb[j];
		}

		/*
		** ====================================================================
		** 2d. TRUE BATCHED FFN (BLAS 3)
		** ====================================================================
		** Previously: per-token GEMV in parallel (weights loaded batch_size times)
		** Now: single GEMM for entire batch (weights loaded ONCE)
		** 
		** Memory access pattern:
		**   OLD: for each token: load W1,W3,W2 -> compute -> next token
		**   NEW: load W1 once -> compute all tokens -> load W3 once -> ...
		** 
		** This is the core insight of efficient batched inference.
		** ====================================================================
		*/

		/*
		** Step 1: Batched RMSNorm for FFN input (AVX2 SIMD - Phase 9)
		** Uses vectorized op_rmsnorm instead of scalar loop
		** Output: batch_xb[batch_size x dim]
		*/
		{
			t_tensor batch_x_ffn, batch_xb_ffn;
			
			/* Setup input tensor [dim x batch_size] */
			batch_x_ffn.data = s->batch_x;
			batch_x_ffn.ndim = 2;
			batch_x_ffn.shape[0] = dim;
			batch_x_ffn.shape[1] = batch_size;
			batch_x_ffn.size = dim * batch_size;
			batch_x_ffn.dtype = DTYPE_F32;
			
			/* Setup output tensor */
			batch_xb_ffn.data = s->batch_xb;
			batch_xb_ffn.ndim = 2;
			batch_xb_ffn.shape[0] = dim;
			batch_xb_ffn.shape[1] = batch_size;
			batch_xb_ffn.size = dim * batch_size;
			batch_xb_ffn.dtype = DTYPE_F32;
			
			/* AVX2 RMSNorm - processes all tokens in batch */
			op_rmsnorm(&batch_xb_ffn, &batch_x_ffn, l->ffn_norm, c->norm_eps);
		}

		/*
		** Step 2: Batched Gate & Up Projections (SINGLE GEMM each!)
		** 
		** Gate: batch_hb[batch_size x hidden_dim] = batch_xb @ W1^T
		** Up:   batch_hb2[batch_size x hidden_dim] = batch_xb @ W3^T
		** 
		** Now we load W1 and W3 only ONCE, not batch_size times!
		*/
		{
			t_tensor batch_xb_t2, batch_hb_t2, batch_hb2_t2;

			/* Setup batch_xb tensor [dim x batch_size] (column-major for matmul) */
			batch_xb_t2.data = s->batch_xb;
			batch_xb_t2.ndim = 2;
			batch_xb_t2.shape[0] = dim;
			batch_xb_t2.shape[1] = batch_size;
			batch_xb_t2.size = dim * batch_size;
			batch_xb_t2.dtype = DTYPE_F32;

			/* Setup batch_hb (gate output) [hidden_dim x batch_size] */
			batch_hb_t2.data = s->batch_hb;
			batch_hb_t2.ndim = 2;
			batch_hb_t2.shape[0] = c->hidden_dim;
			batch_hb_t2.shape[1] = batch_size;
			batch_hb_t2.size = c->hidden_dim * batch_size;
			batch_hb_t2.dtype = DTYPE_F32;

			/* Setup batch_hb2 (up output) [hidden_dim x batch_size] */
			batch_hb2_t2.data = s->batch_hb2;
			batch_hb2_t2.ndim = 2;
			batch_hb2_t2.shape[0] = c->hidden_dim;
			batch_hb2_t2.shape[1] = batch_size;
			batch_hb2_t2.size = c->hidden_dim * batch_size;
			batch_hb2_t2.dtype = DTYPE_F32;

			/* BATCHED GEMM: Gate projection - ONE call for all tokens! */
			op_matmul(&batch_hb_t2, l->w1, &batch_xb_t2);

			/* BATCHED GEMM: Up projection - ONE call for all tokens! */
			op_matmul(&batch_hb2_t2, l->w3, &batch_xb_t2);
		}

		/*
		** Step 3: SwiGLU Activation (Elementwise, Memory-Bound)
		** silu(gate) * up = (gate / (1 + exp(-gate))) * up
		** 
		** This is memory-bound, so OMP parallelization is perfect here.
		** Using fast_expf for 10x speedup over standard expf.
		*/
		{
			int total_elem = batch_size * c->hidden_dim;
			int idx;
			#pragma omp parallel for schedule(static)
			for (idx = 0; idx < total_elem; idx++)
			{
				float g = s->batch_hb[idx];
				float u = s->batch_hb2[idx];
				/* SiLU: g / (1 + exp(-g)) * u */
				float silu = g / (1.0f + fast_expf(-g));
				s->batch_hb[idx] = silu * u;
			}
		}

		/*
		** Step 4: Batched Down Projection (SINGLE GEMM!)
		** 
		** batch_xb[batch_size x dim] = batch_hb @ W2^T
		** 
		** Again: W2 loaded ONCE, reused for all tokens in batch.
		*/
		{
			t_tensor batch_xb_t3, batch_hb_t3;

			/* batch_hb after activation: [hidden_dim x batch_size] */
			batch_hb_t3.data = s->batch_hb;
			batch_hb_t3.ndim = 2;
			batch_hb_t3.shape[0] = c->hidden_dim;
			batch_hb_t3.shape[1] = batch_size;
			batch_hb_t3.size = c->hidden_dim * batch_size;
			batch_hb_t3.dtype = DTYPE_F32;

			/* Output goes to batch_xb: [dim x batch_size] */
			batch_xb_t3.data = s->batch_xb;
			batch_xb_t3.ndim = 2;
			batch_xb_t3.shape[0] = dim;
			batch_xb_t3.shape[1] = batch_size;
			batch_xb_t3.size = dim * batch_size;
			batch_xb_t3.dtype = DTYPE_F32;

			/* BATCHED GEMM: Down projection - ONE call for all tokens! */
			op_matmul(&batch_xb_t3, l->w2, &batch_hb_t3);
		}

		/*
		** Step 5: Residual Connection
		** batch_x += batch_xb (add FFN output to residual stream)
		*/
		{
			int total_elem2 = batch_size * dim;
			int idx2;
			#pragma omp parallel for schedule(static)
			for (idx2 = 0; idx2 < total_elem2; idx2++)
				s->batch_x[idx2] += s->batch_xb[idx2];
		}
	}

	/*
	** CRITICAL: Copy final token state back to s->x for generation phase
	** Without this, transformer_forward() starts with garbage state!
	*/
	if (batch_size > 0)
	{
		int last_token_idx = batch_size - 1;
		memcpy(s->x, s->batch_x + last_token_idx * dim, dim * sizeof(float));
	}
}

