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

	// 1. Embedding
	t_tensor *emb = t->weights.token_embedding;
	if (!emb || !emb->data) {
		exit(1);
	}
	uint16_t *emb_data = (uint16_t *)emb->data;
	uint16_t *token_vec = emb_data + token * c->dim;
	
	// BF16 to F32 conversion
	for (int i = 0; i < c->dim; i++)
	{
		uint32_t val = (uint32_t)token_vec[i] << 16;
		memcpy(&s->x[i], &val, sizeof(float));
	}
	
	// DEBUG: Embedding check (disabled)
	// if (pos == 0) {
	// 	printf("[DEBUG] Embedding for token=%d: x[0..4]=%.4f %.4f %.4f %.4f %.4f\n",
	// 		token, s->x[0], s->x[1], s->x[2], s->x[3], s->x[4]);
	// }

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

		// [FIX] BUFFER ALIASING BUG
		// s->hb is shared between Attention (n_heads*head_dim=4096) and FFN (hidden_dim=9216)
		// Only zero the attention output portion - FFN overwrites its own section
		memset(s->hb, 0, c->n_heads * c->head_dim * sizeof(float));
		
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

		// RoPE
		// printf("L%d RoPE\n", i);
		t_rope_ctx rope_ctx = {
			.head_dim = c->head_dim,
			.theta_base = c->rope_theta,
			.beta_fast = c->beta_fast,
			.beta_slow = c->beta_slow,
			.factor = c->rope_factor,
			// Auto-calculate mscale if 1.0 but factor > 1.0
			// Formula: s = 0.1 * ln(factor) + 1.0
			// We apply sqrt(s) to q and k, so q.k is scaled by s
			.mscale = (c->mscale == 1.0f && c->rope_factor > 1.0f) 
					  ? sqrtf(0.1f * logf(c->rope_factor) + 1.0f) 
					  : c->mscale,
			.orig_ctx = c->orig_ctx_len,
			.thetas_cache = t->rope_thetas  // Use precomputed thetas (eliminates pow() per token)
		};
		

		op_rope(&q_tensor, pos, &rope_ctx);
		op_rope(&k_tensor, pos, &rope_ctx);

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
			// Convert K to F32 for hashing (use first KV head)
			float k_f32[LSH_MAX_DIM];
			for (int d = 0; d < c->head_dim && d < LSH_MAX_DIM; d++)
				k_f32[d] = s->k[d];
			lsh_update(lsh_idx, lsh_ctx, k_f32, c->head_dim, pos);
		}
		
		// ========== KV CACHE EVICTION ==========
		// Auto-evict low-importance keys when cache fills up
		if (s->kv_cache[i].current_seq_len > t->evict_threshold)
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
		
		// Reset attention output buffer
		memset(s->hb, 0, c->n_heads * c->head_dim * sizeof(float));
		
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
			
			topk_indices = arena_alloc(&t->scratch, attend_k * sizeof(int));
			
			t_lsh_ctx *lsh_ctx = (t_lsh_ctx *)t->lsh_ctx;
			t_lsh_index *lsh_idx = (t_lsh_index *)t->lsh_index;
			int head_dim = c->head_dim;
			
			// Allocate candidate positions (window + LSH candidates)
			int max_candidates = kv_len;  // Worst case: all positions
			int *candidate_positions = arena_alloc(&t->scratch, max_candidates * sizeof(int));
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
				// Compute query hash
				float q_mean[LSH_MAX_DIM];
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
			
			// Step 3: Score only candidate positions with SIMD
			float *key_scores = arena_alloc(&t->scratch, n_candidates * sizeof(float));
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
			
			// Step 4: Select Top-K from candidates using MIN-HEAP
			// Complexity: O(n log k) vs O(n*k) for partial sort
			int actual_k = (n_candidates < attend_k) ? n_candidates : attend_k;
			
			// Allocate heap on scratch arena
			t_heap_item *heap = arena_alloc(&t->scratch, actual_k * sizeof(t_heap_item));
			int heap_size = 0;
			
			// Build heap with O(n log k) insertion
			for (int ci = 0; ci < n_candidates; ci++)
				heap_push(heap, &heap_size, actual_k, key_scores[ci], candidate_positions[ci]);
			
			// Extract indices from heap
			heap_extract_indices(heap, heap_size, topk_indices);
			attend_k = heap_size;
		}
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
				t_block_score *scores = arena_alloc(&t->scratch, 
					n_blocks * sizeof(t_block_score));
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
		
		// Residual: x += xb
		for (int j = 0; j < c->dim; j++) s->x[j] += s->xb[j];
		
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
			if (t->nested_learning)
				memcpy(t->fluid_layers[i].hb_cache, s->hb, c->hidden_dim * sizeof(float));
			
			// SIMD+OpenMP optimized adapter matmul: xb += adapter @ hb
			// adapter is [dim x hidden_dim] (BF16), hb is [hidden_dim] (F32)
			t_tensor *adapter = t->fluid_layers[i].w2_weight;
			t_bf16 *a_data = (t_bf16 *)adapter->data;
			int hidden = c->hidden_dim;
			
			#pragma omp parallel for schedule(static)
			for (int row = 0; row < c->dim; row++)
			{
				t_bf16 *a_row = a_data + row * hidden;
				s->xb[row] += simd_dot_bf16_f32(a_row, s->hb, hidden);
			}
		}
		
		// Residual
		for (int j = 0; j < c->dim; j++) s->x[j] += s->xb[j];
		
	}

	// 3. Final Norm
	op_rmsnorm(&x_tensor, &x_tensor, t->weights.norm, c->norm_eps);
	

	// 4. Logits
	wrap_tensor(&logits_tensor, s->logits, c->vocab_size);
	op_matmul(&logits_tensor, t->weights.output, &x_tensor);
	

	return (s->logits);
}

/* NOTE: transformer_backward_step has been moved to src/nested/backward.c
** with FP32 gradient accumulation for mixed precision training.
** See backward_zero_grads() and backward_apply_grads() for new API.
*/
