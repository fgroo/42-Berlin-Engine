#include "inference.h"
#include "../config.h"
#include "compute/ops.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // AVX2 intrinsics
#include <omp.h>        // OpenMP parallelization

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
		// We must scrub stale FFN data from previous layer before each iteration!
		memset(s->hb, 0, c->hidden_dim * sizeof(float));
		
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
			.orig_ctx = c->orig_ctx_len
		};
		

		op_rope(&q_tensor, pos, &rope_ctx);
		op_rope(&k_tensor, pos, &rope_ctx);

		// KV Cache Append
		kv_cache_append(&s->kv_cache[i], &k_tensor, &v_tensor);
		
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
			// SPARSE: Score all keys and select Top-K
			topk_indices = arena_alloc(&t->scratch, attend_k * sizeof(int));
			
			// Score buffer for all keys
			float *key_scores = arena_alloc(&t->scratch, kv_len * sizeof(float));
			t_tensor score_tensor;
			score_tensor.data = key_scores;
			score_tensor.size = kv_len;
			score_tensor.dtype = DTYPE_F32;
			
			// Score each position by cheap dot product with Q
			// (Using average head for speed - full Lightning uses learned weights)
			t_bf16 *k_data = (t_bf16 *)s->kv_cache[i].k.data;
			int kv_stride = c->n_kv_heads * c->head_dim;
			
			for (int ti = 0; ti < kv_len; ti++)
			{
				float score = 0.0f;
				// Sum over KV heads
				for (int kv_h = 0; kv_h < c->n_kv_heads; kv_h++)
				{
					t_bf16 *k_vec = k_data + ti * kv_stride + kv_h * c->head_dim;
					float *q_for_kv = s->q + (kv_h * (c->n_heads/c->n_kv_heads)) * c->head_dim;
					for (int j = 0; j < c->head_dim; j++)
					{
						score += q_for_kv[j] * bf16_to_float(k_vec[j]);
					}
				}
				key_scores[ti] = score;
			}
			
			// Select Top-K indices
			op_topk_select(topk_indices, &score_tensor, attend_k, &t->scratch);
		}
		
		// Multi-head attention loop (PARALLEL)
		// CRITICAL: Zero hb before accumulating attention output!
		memset(s->hb, 0, c->n_heads * c->head_dim * sizeof(float));
		
		#pragma omp parallel for schedule(dynamic, 1)
		for (int h = 0; h < c->n_heads; h++)
		{
			float *q_head = s->q + h * c->head_dim;
			int kv_h = h / (c->n_heads / c->n_kv_heads);
			
			float *scores = s->att + h * c->seq_len;
			float scale = 1.0f / sqrtf(c->head_dim);
			
			t_bf16 *k_data = (t_bf16 *)s->kv_cache[i].k.data;
			t_bf16 *v_data = (t_bf16 *)s->kv_cache[i].v.data;
			int kv_stride = c->n_kv_heads * c->head_dim;
			
			// DEBUG: K cache print (disabled)
			
			// Compute attention scores for selected keys only
			for (int ti = 0; ti < attend_k; ti++)
			{
				int seq_idx = use_sparse ? topk_indices[ti] : ti;
				float score = 0.0f;
				t_bf16 *k_vec = k_data + seq_idx * kv_stride + kv_h * c->head_dim;
				
				for (int j = 0; j < c->head_dim; j++)
				{
					score += q_head[j] * bf16_to_float(k_vec[j]);
				}
				scores[ti] = score * scale;
			}
			

			
			// Softmax over attend_k (not kv_len!)
			float max_val = -INFINITY;
			for (int ti = 0; ti < attend_k; ti++)
				if (scores[ti] > max_val) max_val = scores[ti];
			
			float sum_exp = 0.0f;
			for (int ti = 0; ti < attend_k; ti++)
			{
				scores[ti] = expf(scores[ti] - max_val);
				sum_exp += scores[ti];
			}
			
			float inv_sum = 1.0f / sum_exp;
			for (int ti = 0; ti < attend_k; ti++)
				scores[ti] *= inv_sum;
			

			
			// Weighted sum over selected keys only
			float *out_head = s->hb + h * c->head_dim;
			
			for (int ti = 0; ti < attend_k; ti++)
			{
				int seq_idx = use_sparse ? topk_indices[ti] : ti;
				float weight = scores[ti];
				t_bf16 *v_vec = v_data + seq_idx * kv_stride + kv_h * c->head_dim;
				
				for (int j = 0; j < c->head_dim; j++)
				{
					out_head[j] += weight * bf16_to_float(v_vec[j]);
				}
			}
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
		if (t->nested_learning && t->fluid_layers)
		{
			// CRITICAL: Cache hb for backward pass (each layer needs its own)
			memcpy(t->fluid_layers[i].hb_cache, s->hb, c->hidden_dim * sizeof(float));
			
			t_tensor *adapter = t->fluid_layers[i].w2_weight;
			t_bf16 *a_data = (t_bf16 *)adapter->data;
			// Naive matmul: adapter is [dim x hidden_dim], hb is [hidden_dim]
			for (int row = 0; row < c->dim; row++)
			{
				float sum = 0.0f;
				for (int col = 0; col < c->hidden_dim; col++)
				{
					sum += bf16_to_float(a_data[row * c->hidden_dim + col]) 
					     * s->hb[col];
				}
				s->xb[row] += sum;
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

// ==================== NESTED LEARNING BACKWARD ====================
// Compute gradient of cross-entropy loss w.r.t. fluid adapter weights
// Called after forward pass with the actual target token
void transformer_backward_step(t_transformer *t, int target_token, int pos)
{
	(void)pos; // Currently unused, may be needed for future positional gradients
	
	if (!t->nested_learning || !t->fluid_layers)
		return;
	
	t_transformer_config *c = &t->config;
	t_inference_state *s = &t->state;
	
	// 1. Compute softmax probabilities from logits
	float max_logit = s->logits[0];
	for (int i = 1; i < c->vocab_size; i++)
		if (s->logits[i] > max_logit) max_logit = s->logits[i];
	
	float sum_exp = 0.0f;
	for (int i = 0; i < c->vocab_size; i++)
	{
		s->logits[i] = expf(s->logits[i] - max_logit);
		sum_exp += s->logits[i];
	}
	
	// Softmax and gradient of cross-entropy in one pass
	// grad = softmax(logits) - one_hot(target)
	float inv_sum = 1.0f / sum_exp;
	float target_prob = 0.0f;
	for (int i = 0; i < c->vocab_size; i++)
		s->logits[i] = s->logits[i] * inv_sum; // Now softmax probs
	target_prob = s->logits[target_token]; // Probability of correct token
	s->logits[target_token] -= 1.0f; // Subtract 1 at target position
	
	// Cross-entropy loss = -log(target_prob)
	float loss = -logf(target_prob + 1e-8f);
	static int nl_step = 0;
	static int skipped = 0;
	
	// SURPRISE-BASED LEARNING: Only learn from "surprising" tokens
	// If loss < threshold, the model already knew this - skip expensive backward pass
	if (loss < LEARNING_THRESHOLD) {
		skipped++;
		if (nl_step < 5 || nl_step % 20 == 0) {
			printf("[NL] Step %d: Loss=%.2f (SKIP - below threshold) [skipped %d]\n", 
				nl_step, loss, skipped);
			fflush(stdout);
		}
		nl_step++;
		return;  // Skip expensive gradient computation!
	}
	
	if (nl_step < 5 || nl_step % 20 == 0) {
		printf("[NL] Step %d: Loss=%.2f, P(target)=%.1f%% [LEARN]\n", nl_step, loss, target_prob * 100);
		fflush(stdout);
	}
	nl_step++;
	
	// 2. Backprop through output layer to get grad_x
	// grad_x = output_weight^T @ grad_logits
	// Using pre-transposed weights for row-major (cache-friendly) access
	memset(s->grad_x, 0, c->dim * sizeof(float));
	
	if (t->output_weight_T)
	{
		// FAST PATH: Use pre-transposed BF16 weights with SIMD
		// Layout: output_weight_T[d * vocab + v]
		int vocab = c->vocab_size;
		
		#pragma omp parallel for schedule(static)
		for (int d = 0; d < c->dim; d++)
		{
			t_bf16 *row = t->output_weight_T + d * vocab;
			__m256 sum_vec = _mm256_setzero_ps();
			int v = 0;
			
			// Process 8 elements at a time with AVX2
			for (; v + 7 < vocab; v += 8)
			{
				// Load 8 BF16 weights, convert to F32
				__m128i bf16_w = _mm_loadu_si128((__m128i *)(row + v));
				__m256i w_32 = _mm256_cvtepu16_epi32(bf16_w);
				w_32 = _mm256_slli_epi32(w_32, 16);
				__m256 w_f32 = _mm256_castsi256_ps(w_32);
				
				// Load 8 F32 logits (grad_logits are in s->logits after softmax)
				__m256 l_vals = _mm256_loadu_ps(s->logits + v);
				
				// FMA: sum += w * logits
				sum_vec = _mm256_fmadd_ps(w_f32, l_vals, sum_vec);
			}
			
			// Horizontal sum of sum_vec
			__m128 lo = _mm256_castps256_ps128(sum_vec);
			__m128 hi = _mm256_extractf128_ps(sum_vec, 1);
			lo = _mm_add_ps(lo, hi);
			lo = _mm_hadd_ps(lo, lo);
			lo = _mm_hadd_ps(lo, lo);
			float sum = _mm_cvtss_f32(lo);
			
			// Scalar remainder
			for (; v < vocab; v++)
				sum += bf16_to_float(row[v]) * s->logits[v];
			
			s->grad_x[d] = sum;
		}
	}
	else
	{
		// FALLBACK: Naive column-major access (slow, but works without transpose)
		t_bf16 *ow_data = (t_bf16 *)t->weights.output->data;
		for (int d = 0; d < c->dim; d++)
		{
			float sum = 0.0f;
			for (int v = 0; v < c->vocab_size; v++)
				sum += bf16_to_float(ow_data[v * c->dim + d]) * s->logits[v];
			s->grad_x[d] = sum;
		}
	}
	
	// 3. Update each layer's fluid adapter (reverse order for proper backprop)
	// Use cached hb_cache per layer (fixed cache miss bug!)
	// LAYER FREEZING: Skip lower layers to reduce bandwidth, upper layers are smarter
	
	for (int layer = c->n_layers - 1; layer >= 0; layer--)
	{
		// Skip frozen layers (lower layers typically have less task-specific knowledge)
		if (layer < FROZEN_LAYERS)
			continue ;
		
		t_fluid_layer *fl = &t->fluid_layers[layer];
		t_bf16 *w_data = (t_bf16 *)fl->w2_weight->data;
		float *hb_cached = fl->hb_cache;
		float lr = t->nested_lr;
		
		// SIMD-optimized weight update with gradient clipping (PARALLEL)
		// Process 8 elements at a time with AVX2
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < c->dim; i++)
		{
			float grad_xi = s->grad_x[i];
			
			// GRADIENT CLIPPING: Prevent exploding gradients
			if (grad_xi > GRADIENT_CLIP) grad_xi = GRADIENT_CLIP;
			if (grad_xi < -GRADIENT_CLIP) grad_xi = -GRADIENT_CLIP;
			
			__m256 v_grad_xi = _mm256_set1_ps(grad_xi);
			__m256 v_lr = _mm256_set1_ps(lr);
			__m256 v_clip_max = _mm256_set1_ps(1.0f);
			__m256 v_clip_min = _mm256_set1_ps(-1.0f);
			
			int j = 0;
			int row_offset = i * c->hidden_dim;
			
			// AVX2 loop: process 8 floats at a time
			for (; j + 7 < c->hidden_dim; j += 8)
			{
				int idx = row_offset + j;
				
				// Load 8 hb values (already float)
				__m256 v_hb = _mm256_loadu_ps(&hb_cached[j]);
				
				// Compute gradient: g = grad_x[i] * hb[j]
				__m256 v_g = _mm256_mul_ps(v_grad_xi, v_hb);
				
				// Clip gradient to [-1, 1]
				v_g = _mm256_min_ps(v_g, v_clip_max);
				v_g = _mm256_max_ps(v_g, v_clip_min);
				
				// Load 8 BF16 weights, convert to FP32
				// BF16 -> FP32: shift left by 16 bits
				__m128i bf16_w = _mm_loadu_si128((__m128i*)&w_data[idx]);
				__m256i w_32 = _mm256_cvtepu16_epi32(bf16_w);
				w_32 = _mm256_slli_epi32(w_32, 16);  // BF16 in upper 16 bits
				__m256 v_w = _mm256_castsi256_ps(w_32);
				
				// SGD update: w = w - lr * g
				__m256 v_update = _mm256_fnmadd_ps(v_lr, v_g, v_w);
				
				// Convert FP32 back to BF16 with ROUNDING (not truncation!)
				// Truncation causes gradient drift: 1.99 -> 1, but rounding: 1.99 -> 2
				// Add 0x8000 (0.5 in the lower 16-bit word) to round to nearest
				__m256i result_32 = _mm256_castps_si256(v_update);
				__m256i rounding = _mm256_set1_epi32(0x8000);
				result_32 = _mm256_add_epi32(result_32, rounding);
				result_32 = _mm256_srli_epi32(result_32, 16);
				// Pack 32-bit to 16-bit (only lower 16 bits of each 32-bit element)
				__m128i result_16 = _mm256_cvtepi32_epi16_custom(result_32);
				_mm_storeu_si128((__m128i*)&w_data[idx], result_16);
			}
			
			// Scalar fallback for remaining elements
			for (; j < c->hidden_dim; j++)
			{
				float g = grad_xi * hb_cached[j];
				// Clip gradient
				if (g > 1.0f) g = 1.0f;
				if (g < -1.0f) g = -1.0f;
				int idx = row_offset + j;
				float w = bf16_to_float(w_data[idx]);
				w_data[idx] = float_to_bf16(w - lr * g);
			}
		}
	}
}
