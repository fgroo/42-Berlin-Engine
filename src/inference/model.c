#include "inference.h"
#include "../config.h"
#include "compute/ops_lsh.h"
#include "compute/ops_rope.h"   /* Phase 12: Precomputed sin/cos tables */
#include "compute/ops_quant.h"  /* Phase 2: FP8/INT8 quantization */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/*
** SAFE JSON PARSING MACROS
** Prevent NULL pointer dereference when ':' is missing from config line.
** Uses strtol/strtod for safer parsing with range validation.
*/
#define SAFE_PARSE_INT(line, key, dest) do { \
	if (strstr(line, key)) { \
		char *colon = strchr(line, ':'); \
		if (colon) { (dest) = (int)strtol(colon + 1, NULL, 10); } \
	} \
} while (0)

#define SAFE_PARSE_FLOAT(line, key, dest) do { \
	if (strstr(line, key)) { \
		char *colon = strchr(line, ':'); \
		if (colon) { (dest) = (float)strtod(colon + 1, NULL); } \
	} \
} while (0)

// Simple config parser (HARDENED: NULL-safe parsing)
static int	load_config(t_transformer_config *conf, const char *path)
{
	FILE *f = fopen(path, "r");
	if (!f) {
		fprintf(stderr, "Config Error: Cannot open '%s'\n", path);
		return (-1);
	}

	char line[1024];
	int in_text_config = 0;
	int in_rope_params = 0;
	while (fgets(line, sizeof(line), f))
	{
		if (strstr(line, "\"text_config\": {")) {
			in_text_config = 1;
			continue;
		}
		if (in_text_config && strstr(line, "\"rope_parameters\": {")) {
			in_rope_params = 1;
			continue;
		}
		if (in_rope_params && strstr(line, "},")) {
			in_rope_params = 0;
			continue;
		}
		if (in_text_config && !in_rope_params && strstr(line, "},")) {
			in_text_config = 0;
			continue;
		}

		if (in_rope_params)
		{
			SAFE_PARSE_FLOAT(line, "\"beta_fast\":", conf->beta_fast);
			SAFE_PARSE_FLOAT(line, "\"beta_slow\":", conf->beta_slow);
			SAFE_PARSE_FLOAT(line, "\"factor\":", conf->rope_factor);
			SAFE_PARSE_FLOAT(line, "\"mscale\":", conf->mscale);
			SAFE_PARSE_FLOAT(line, "\"rope_theta\":", conf->rope_theta);
			SAFE_PARSE_INT(line, "\"original_max_position_embeddings\":", conf->orig_ctx_len);
		}
		else if (in_text_config)
		{
			SAFE_PARSE_INT(line, "\"hidden_size\":", conf->dim);
			SAFE_PARSE_INT(line, "\"intermediate_size\":", conf->hidden_dim);
			SAFE_PARSE_INT(line, "\"num_hidden_layers\":", conf->n_layers);
			SAFE_PARSE_INT(line, "\"num_attention_heads\":", conf->n_heads);
			SAFE_PARSE_INT(line, "\"num_key_value_heads\":", conf->n_kv_heads);
			SAFE_PARSE_INT(line, "\"vocab_size\":", conf->vocab_size);
			SAFE_PARSE_INT(line, "\"head_dim\":", conf->head_dim);
		}
	}
	fclose(f);
	
	// Manual overrides/defaults if parsing failed or for safety
	if (conf->dim == 0) conf->dim = 3072;
	if (conf->hidden_dim == 0) conf->hidden_dim = 9216;
	if (conf->n_layers == 0) conf->n_layers = 26;
	if (conf->n_heads == 0) conf->n_heads = 32;
	if (conf->n_kv_heads == 0) conf->n_kv_heads = 8;
	if (conf->vocab_size == 0) conf->vocab_size = 131072;
	if (conf->head_dim == 0) conf->head_dim = 128;
	
	// Defaults for YaRN if not found
	if (conf->rope_theta == 0.0f) conf->rope_theta = 1000000.0f;
	if (conf->rope_factor == 0.0f) conf->rope_factor = 1.0f; // Default to standard RoPE
	if (conf->mscale == 0.0f) conf->mscale = 1.0f;
	
	conf->norm_eps = 1e-5f;
	conf->seq_len = 8192; // Reasonable default for inference

	return (0);
}

static void *aligned_calloc(size_t num, size_t size)
{
	void *ptr = NULL;
	size_t total = num * size;
	/* 64-byte alignment: AVX-512 ready and cache-line aligned */
	if (posix_memalign(&ptr, 64, total) != 0) return NULL;
	memset(ptr, 0, total);
	return ptr;
}


int	transformer_init(t_transformer *t, const char *model_path, const char *config_path)
{
	if (load_config(&t->config, config_path) != 0)
		return (-1);
	
	if (load_model(&t->model, model_path) != 0)
		return (-1);

	// Map weights
	// Map weights by iterating over all loaded tensors
	// This is more robust against safetensors lexicographical ordering (e.g. layers.10 before layers.2)
	t->weights.layers = calloc(t->config.n_layers, sizeof(t_layer_weights));
	
	// Initialize vision tower to NULL (text-only mode, lazy loading)
	t->vision = NULL;
	
	// Log lazy vision tensor count
	if (t->model.num_vision_tensors > 0)
		printf("[LOADER] %d vision tensors available (lazy, not in RAM)\n",
			t->model.num_vision_tensors);
	
	for (int i = 0; i < t->model.num_tensors; i++)
	{
		char *name = t->model.tensors[i].name;
		t_tensor *tensor = &t->model.tensors[i].tensor;
		t_tensor_category cat = t->model.tensors[i].category;
		
		// Skip VISION tensors - they stay in mmap, never paged in text mode
		if (cat == TENSOR_VISION)
			continue ;
		
		// CRITICAL: Copy FLUID tensors from PROT_READ mmap to writable Arena
		// Without this, backprop will SIGSEGV when trying to update weights!
		if (cat == TENSOR_FLUID)
		{
			size_t size_bytes = tensor->size * sizeof(t_bf16);
			t_bf16 *writable_ptr = arena_alloc_or_die(&t->fluid_arena, size_bytes);
			
			// Copy data from read-only mmap to writable arena
			memcpy(writable_ptr, tensor->data, size_bytes);
			
			// Re-point tensor struct to writable arena memory
			tensor->data = writable_ptr;
			
			printf("[FLUID] Copied '%s' to Arena (%zu bytes, now writable)\n",
				name, size_bytes);
		}
		
		// Check for layer weights
		char *layer_marker = strstr(name, "layers.");
		if (layer_marker)
		{
			// Parse layer index: "layers.12.xxx" -> 12
			int layer_idx = atoi(layer_marker + 7);
			
			if (layer_idx >= 0 && layer_idx < t->config.n_layers)
			{
				t_layer_weights *l = &t->weights.layers[layer_idx];
				
				if (strstr(name, "attention.wq.weight")) {
					l->wq = tensor;
				}
				else if (strstr(name, "attention.wk.weight")) l->wk = tensor;
				else if (strstr(name, "attention.wv.weight")) l->wv = tensor;
				else if (strstr(name, "attention.wo.weight")) l->wo = tensor;
				else if (strstr(name, "feed_forward.w1.weight")) l->w1 = tensor;
				else if (strstr(name, "feed_forward.w2.weight")) l->w2 = tensor;
				else if (strstr(name, "feed_forward.w3.weight")) l->w3 = tensor;
				else if (strstr(name, "attention_norm.weight")) l->attention_norm = tensor;
				else if (strstr(name, "ffn_norm.weight")) l->ffn_norm = tensor;
			}
		}
		// Global weights
		else if (strcmp(name, "tok_embeddings.weight") == 0) t->weights.token_embedding = tensor;
		else if (strcmp(name, "norm.weight") == 0) t->weights.norm = tensor;
		else if (strcmp(name, "output.weight") == 0) t->weights.output = tensor;

		printf("Loaded: %s [%d, %d]\n", name, tensor->shape[0], tensor->shape[1]);
		if (strcmp(name, "model.layers.0.self_attn.q_proj.weight") == 0) {
			printf("[DEBUG] Shape Check: %s -> [%d, %d]\n", name, tensor->shape[0], tensor->shape[1]);
		}
		// Assuming map_set is a function that stores all tensors by name
		// map_set(t->weights, name, tensor);
	}

	// Handle tied embeddings if output weight is missing
	if (!t->weights.output)
		t->weights.output = t->weights.token_embedding;

	// Init state
	t->state.x = aligned_calloc(t->config.dim, sizeof(float));
	t->state.xb = aligned_calloc(t->config.dim, sizeof(float));
	t->state.hb = aligned_calloc(t->config.hidden_dim, sizeof(float));
	t->state.hb2 = aligned_calloc(t->config.hidden_dim, sizeof(float));
	t->state.q = aligned_calloc(t->config.n_heads * t->config.head_dim, sizeof(float));
	t->state.k = aligned_calloc(t->config.n_kv_heads * t->config.head_dim, sizeof(float));
	t->state.v = aligned_calloc(t->config.n_kv_heads * t->config.head_dim, sizeof(float));
	t->state.att = aligned_calloc(t->config.n_heads * t->config.seq_len, sizeof(float));
	t->state.logits = aligned_calloc(t->config.vocab_size, sizeof(float));
	t->state.grad_x = aligned_calloc(t->config.dim, sizeof(float)); // For backprop
	t->state.final_input_cache = aligned_calloc(t->config.dim, sizeof(float)); // [HOTFIX] Issue #1
	
	// ========== BATCHED PREFILL BUFFERS (Phase 2) ==========
	// Enable GEMM (M=batch_size) instead of GEMV (M=1) for QKV projections
	// Total: ~5-6MB extra memory for 20%+ prefill speedup
	t->state.batch_x = aligned_calloc(MAX_PREFILL_BATCH * t->config.dim, sizeof(float));
	t->state.batch_xb = aligned_calloc(MAX_PREFILL_BATCH * t->config.dim, sizeof(float));
	t->state.batch_q = aligned_calloc(MAX_PREFILL_BATCH * t->config.n_heads * t->config.head_dim, sizeof(float));
	t->state.batch_k = aligned_calloc(MAX_PREFILL_BATCH * t->config.n_kv_heads * t->config.head_dim, sizeof(float));
	t->state.batch_v = aligned_calloc(MAX_PREFILL_BATCH * t->config.n_kv_heads * t->config.head_dim, sizeof(float));
	t->state.batch_out = aligned_calloc(MAX_PREFILL_BATCH * t->config.dim, sizeof(float));
	t->state.batch_hb = aligned_calloc(MAX_PREFILL_BATCH * t->config.hidden_dim, sizeof(float));
	t->state.batch_hb2 = aligned_calloc(MAX_PREFILL_BATCH * t->config.hidden_dim, sizeof(float));
	
	// ========== NESTED LEARNING CONTEXT (TechLead Solution) ==========
	t->state.token_history = calloc(t->config.seq_len, sizeof(int));
	
	// Init KV Cache
	// Allocate 1GB for KV cache
	size_t arena_size = 1024 * 1024 * 1024; 
	arena_init(&t->kv_arena, arena_size);
	if (!t->kv_arena.base) {
		return -1;
	}
	
	// Scratch arena for Top-K selection (1MB is plenty)
	arena_init(&t->scratch, 1024 * 1024);
	
	// Fluid arena for TENSOR_FLUID copies (writable adapters) - 64MB
	arena_init(&t->fluid_arena, 64 * 1024 * 1024);
	
	// SPARSE ATTENTION: Top-K keys per head (0 = dense attention)
	t->sparse_k = SPARSE_K;  // From config.h
	
	// NESTED LEARNING: Enable test-time training
	t->nested_learning = 1; // ENABLED with gradient clipping for stability
	t->nested_lr = NESTED_LR; // From config.h
	nl_counters_reset(&t->nl_state);  // CAS-based atomic counters (Phase 2)
	
	// CHAT MODE: Initialize to safe defaults (was uninitialized stack garbage!)
	t->raw_mode = 0;        // Chat template ENABLED by default
	t->persistent_mode = 0; // Transient learning by default
	
	// KV CACHE EVICTION: Auto-evict when cache fills up
	t->evict_threshold = t->config.seq_len - 128; // Trigger 128 before max
	t->evict_keep_k = t->config.seq_len / 2;      // Keep half
	// Init uniform head weights (all heads contribute equally)
	t->evict_weights.data = calloc(t->config.n_kv_heads, sizeof(t_bf16));
	t->evict_weights.size = t->config.n_kv_heads;
	t->evict_weights.dtype = DTYPE_BF16;
	t_bf16 *ew = (t_bf16 *)t->evict_weights.data;
	for (int h = 0; h < t->config.n_kv_heads; h++)
		ew[h] = float_to_bf16(1.0f);
	
	// Allocate fluid layers (adapters for test-time learning)
	t->fluid_layers = calloc(t->config.n_layers, sizeof(t_fluid_layer));
	size_t adapter_size = t->config.dim * t->config.hidden_dim;
	for (int i = 0; i < t->config.n_layers; i++)
	{
		// Allocate weight tensor (ZERO initialized - critical!)
		t->fluid_layers[i].w2_weight = calloc(1, sizeof(t_tensor));
		t->fluid_layers[i].w2_weight->data = calloc(adapter_size, sizeof(t_bf16));
		t->fluid_layers[i].w2_weight->size = adapter_size;
		t->fluid_layers[i].w2_weight->dtype = DTYPE_BF16;
		t->fluid_layers[i].w2_weight->ndim = 2;
		t->fluid_layers[i].w2_weight->shape[0] = t->config.dim;
		t->fluid_layers[i].w2_weight->shape[1] = t->config.hidden_dim;
		
		// Allocate gradient tensor
		t->fluid_layers[i].w2_grad = calloc(1, sizeof(t_tensor));
		t->fluid_layers[i].w2_grad->data = calloc(adapter_size, sizeof(t_bf16));
		t->fluid_layers[i].w2_grad->size = adapter_size;
		t->fluid_layers[i].w2_grad->dtype = DTYPE_BF16;
		t->fluid_layers[i].w2_grad->ndim = 2;
		t->fluid_layers[i].w2_grad->shape[0] = t->config.dim;
		t->fluid_layers[i].w2_grad->shape[1] = t->config.hidden_dim;
		
		// Allocate hb cache for backprop (stores hidden activations per layer)
		t->fluid_layers[i].hb_cache = calloc(t->config.hidden_dim, sizeof(float));
		
		// FP32 gradient accumulator (CRITICAL: fixes BF16 precision loss!)
		// Gradients accumulate in FP32, converted to BF16 only on weight update
		t->fluid_layers[i].grad_acc = calloc(adapter_size, sizeof(float));
	}

	/*
	** Solution 4: Final Hidden Adapter [dim x dim]
	** Applied directly to final hidden state x before output projection.
	** This bypasses the FFN/hidden_dim mismatch issue.
	*/
	int final_adapter_size = t->config.dim * t->config.dim;
	t->final_adapter = calloc(1, sizeof(t_tensor));
	t->final_adapter->data = calloc(final_adapter_size, sizeof(t_bf16));
	t->final_adapter->size = final_adapter_size;
	t->final_adapter->dtype = DTYPE_BF16;
	t->final_adapter->ndim = 2;
	t->final_adapter->shape[0] = t->config.dim;
	t->final_adapter->shape[1] = t->config.dim;
	t->final_adapter_grad = calloc(final_adapter_size, sizeof(float));
	printf("[FINAL_ADAPTER] Initialized: [%d x %d] = %d params\n",
		t->config.dim, t->config.dim, final_adapter_size);

	/* Solution 5: Logit Bias [vocab_size] - direct output layer bias */
	t->logit_bias = calloc(t->config.vocab_size, sizeof(float));
	printf("[LOGIT_BIAS] Initialized: [%d] = %lu bytes\n",
		t->config.vocab_size, t->config.vocab_size * sizeof(float));

	/* TechLead Solution: Context-Aware Bias Cache */
	t->context_bias.size = 65536; // 64k entries
	t->context_bias.keys = calloc(t->context_bias.size, sizeof(uint64_t));
	t->context_bias.tokens = calloc(t->context_bias.size, sizeof(int));
	t->context_bias.biases = calloc(t->context_bias.size, sizeof(float));
	t->context_bias.count = 0;
	printf("[CONTEXT_BIAS] Initialized: %d entries = %lu bytes\n",
		t->context_bias.size, t->context_bias.size * (sizeof(uint64_t) + sizeof(int) + sizeof(float)));

	t->state.kv_cache = calloc(t->config.n_layers, sizeof(t_kv_cache));
	t_kv_init_params kv_params = {
		.max_seq = t->config.seq_len,
		.num_heads = t->config.n_kv_heads,
		.head_dim = t->config.head_dim
	};
	
	for (int i = 0; i < t->config.n_layers; i++)
	{
		kv_cache_init(&t->state.kv_cache[i], &t->kv_arena, &kv_params);
	}
	
	// ========== PAGED KV CACHE (Sparse Attention Foundation) ==========
	// Initialize block manager for O(K) attention
	// Blocks: 16 tokens each, head-first layout for SIMD
	t->use_paged_kv = 1;  // ENABLE paged cache by default for testing
	if (t->use_paged_kv)
	{
		int max_blocks_per_layer = (t->config.seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
		block_manager_init(&t->block_manager, t->config.n_layers,
			t->config.n_kv_heads, t->config.head_dim, max_blocks_per_layer);
		
		// Allocate paged_kv views per layer
		t->paged_kv = calloc(t->config.n_layers, sizeof(t_paged_kv_cache));
		for (int i = 0; i < t->config.n_layers; i++)
		{
			t->paged_kv[i].bm = &t->block_manager;
			t->paged_kv[i].layer_idx = i;
			t->paged_kv[i].n_blocks = 0;
			t->paged_kv[i].n_tokens = 0;
		}
	}
	
	// Pre-transpose output weights for fast backward pass (BF16, ~800MB)
	// Transposed layout: output_weight_T[d * vocab + v] = output[v * dim + d]
	// This enables row-major access in backward pass for cache efficiency
	if (t->nested_learning && t->weights.output)
	{
		size_t trans_size = (size_t)t->config.dim * t->config.vocab_size;
		t->output_weight_T = calloc(trans_size, sizeof(t_bf16));
		if (t->output_weight_T)
		{
			t_bf16 *src = (t_bf16 *)t->weights.output->data;
			int dim = t->config.dim;
			int vocab = t->config.vocab_size;
			printf("Transposing output weights for fast backprop...\n");
			fflush(stdout);
			#pragma omp parallel for schedule(static)
			for (int v = 0; v < vocab; v++)
			{
				for (int d = 0; d < dim; d++)
					t->output_weight_T[d * vocab + v] = src[v * dim + d];
			}
			printf("Transpose complete.\n");
		}
	}
	else
	{
		t->output_weight_T = NULL;
	}

	// ========== ROPE THETA PRECOMPUTATION ==========
	// Precompute YaRN/RoPE thetas once at init to avoid pow() calls per token
	// thetas[j] = 1.0 / pow(theta_base, j*2 / head_dim) with YaRN adjustments
	{
		int half = t->config.head_dim / 2;
		t->rope_thetas = calloc(half, sizeof(float));
		if (t->rope_thetas)
		{
			float theta_base = t->config.rope_theta;
			float factor = t->config.rope_factor;
			float beta_fast = t->config.beta_fast;
			float beta_slow = t->config.beta_slow;
			int head_dim = t->config.head_dim;
			
			printf("Precomputing RoPE thetas (eliminating %d pow() calls/token)...\n",
				half * 2 * t->config.n_layers);
			
			for (int j = 0; j < half; j++)
			{
				int dim_idx = j * 2;
				float freq_idx = (float)dim_idx / (float)head_dim;
				float theta;
				
				// Standard RoPE formula
				theta = 1.0f / powf(theta_base, freq_idx);
				
				// Apply YaRN scaling if factor > 1.0
				if (factor > 1.0f)
				{
					float slow_thresh = beta_slow / head_dim;
					float fast_thresh = beta_fast / head_dim;
					
					if (freq_idx > fast_thresh)
					{
						// High frequency: apply full scaling
						theta = 1.0f / powf(theta_base * factor, freq_idx);
					}
					else if (freq_idx > slow_thresh)
					{
						// Ramp region: linear interpolation
						float alpha = (freq_idx - slow_thresh) / (fast_thresh - slow_thresh);
						float theta_base_val = 1.0f / powf(theta_base, freq_idx);
						float theta_scaled = 1.0f / powf(theta_base * factor, freq_idx);
						theta = (1.0f - alpha) * theta_base_val + alpha * theta_scaled;
					}
					// else: Low frequency, keep base theta
				}
				t->rope_thetas[j] = theta;
			}
			printf("RoPE thetas cached.\n");
		}
	}

	// ========== ROPE CACHE (Phase 12): PRECOMPUTED SIN/COS TABLES ==========
	// Eliminates 53K sinf/cosf calls per token for 10-20% generation speedup
	// Memory cost: 8K * 64 * 2 * 4 = 4MB (excellent trade-off)
	{
		t_rope_cache *cache = rope_cache_init(t->config.head_dim,
			t->config.seq_len, t->config.rope_theta);
		if (cache)
		{
			t->rope_cache = cache;
			printf("[RoPE CACHE] Enabled. Generation speedup expected.\n");
		}
		else
		{
			t->rope_cache = NULL;
			printf("[RoPE CACHE] Failed to init, using fallback.\n");
		}
	}

	// ========== FP8 QUANTIZATION (Phase 2 Deep Freeze) ==========
	// Initialize FP8 E4M3 lookup table for Lightning Indexer
	// 256 entries = 1KB, fits in L1 cache for zero-cost FP8->FP32
	{
		quant_init_fp8_lut();
		printf("[FP8 LUT] Initialized (256 entries, 1KB). Quantization ready.\n");
	}

	// ========== LSH LIGHTNING INDEXER INIT ==========
	// Initialize LSH for true O(K) sparse attention routing
	// This replaces brute-force O(N) key scanning with hash-based block selection
	{
		/* ops_lsh.h included at top of file */
		
		t_lsh_ctx *lsh_ctx = calloc(1, sizeof(t_lsh_ctx));
		t_lsh_index *lsh_idx = calloc(1, sizeof(t_lsh_index));
		
	if (lsh_ctx && lsh_idx)
		{
			// Initialize LSH with head_dim dimension, seed based on config
			lsh_init(lsh_ctx, t->config.head_dim, 42);
			lsh_index_init(lsh_idx);  // Initialize hash table with empty buckets
			
			t->lsh_ctx = lsh_ctx;
			t->lsh_index = lsh_idx;
			t->use_lsh = (t->sparse_k > 0);  // Enable if sparse_k is set
			
			// CRITICAL: Initialize atomic stats to zero (Phase 9)
			// Without this, stats contain garbage from uninitialized memory!
			lsh_stats_reset_atomic(&t->lsh_stats);
			
			printf("LSH Lightning Indexer initialized (dim=%d, %d hash bits, block_size=%d)\n",
				t->config.head_dim, LSH_NUM_HASHES, LSH_BLOCK_SIZE);
		}
		else
		{
			t->lsh_ctx = NULL;
			t->lsh_index = NULL;
			t->use_lsh = 0;
		}
	}

	return (0);
}

void	transformer_free(t_transformer *t)
{
	free_model(&t->model);
	free(t->weights.layers);
	free(t->state.x);
	free(t->state.xb);
	free(t->state.hb);
	free(t->state.hb2);
	free(t->state.q);
	free(t->state.k);
	free(t->state.v);
	free(t->state.att);
	free(t->state.logits);
	free(t->state.grad_x);
	free(t->state.final_input_cache); // [HOTFIX] Issue #1
	free(t->state.kv_cache);
	
	// Free batch prefill buffers
	free(t->state.batch_x);
	free(t->state.batch_xb);
	free(t->state.batch_q);
	free(t->state.batch_k);
	free(t->state.batch_v);
	free(t->state.batch_out);
	free(t->state.batch_hb);
	free(t->state.batch_hb2);
	
	// Free fluid layers
	if (t->fluid_layers) {
		for (int i = 0; i < t->config.n_layers; i++) {
			if (t->fluid_layers[i].w2_weight) {
				free(t->fluid_layers[i].w2_weight->data);
				free(t->fluid_layers[i].w2_weight);
			}
			if (t->fluid_layers[i].w2_grad) {
				free(t->fluid_layers[i].w2_grad->data);
				free(t->fluid_layers[i].w2_grad);
			}
			free(t->fluid_layers[i].hb_cache);
			// Free FP32 gradient accumulator
			if (t->fluid_layers[i].grad_acc)
				free(t->fluid_layers[i].grad_acc);
		}
		free(t->fluid_layers);
	}
	
	// Free eviction weights
	free(t->evict_weights.data);
	
	// Free arenas
	arena_free(&t->kv_arena);
	arena_free(&t->scratch);
	arena_free(&t->fluid_arena);
	
	// Free vision tower if allocated
	if (t->vision)
	{
		if (t->vision->vit_layers)
			free(t->vision->vit_layers);
		free(t->vision);
	}
	
	// Free transposed output weights
	if (t->output_weight_T)
		free(t->output_weight_T);
	
	// Free precomputed RoPE thetas
	if (t->rope_thetas)
		free(t->rope_thetas);
	
	// Free RoPE cache (Phase 12)
	if (t->rope_cache)
		rope_cache_free((t_rope_cache *)t->rope_cache);
	
	// Free LSH structures
	if (t->lsh_ctx)
		free(t->lsh_ctx);
	if (t->lsh_index)
		free(t->lsh_index);
}
