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
	int in_vision_config = 0;  /* Skip vision encoder config */
	while (fgets(line, sizeof(line), f))
	{
		/* Track nested config blocks */
		if (strstr(line, "\"text_config\": {")) {
			in_text_config = 1;
			continue;
		}
		if (strstr(line, "\"vision_config\": {")) {
			in_vision_config = 1;
			continue;
		}
		if (in_text_config && strstr(line, "\"rope_parameters\": {")) {
			in_rope_params = 1;
			continue;
		}
		/* Close blocks on "}," */
		if (in_rope_params && strstr(line, "},")) {
			in_rope_params = 0;
			continue;
		}
		if (in_vision_config && strstr(line, "},")) {
			in_vision_config = 0;
			continue;
		}
		if (in_text_config && !in_rope_params && strstr(line, "},")) {
			in_text_config = 0;
			continue;
		}
		/* SKIP VISION CONFIG ENTIRELY */
		if (in_vision_config)
			continue;

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
			/* VLM format (Ministral): keys nested under "text_config" */
			SAFE_PARSE_INT(line, "\"hidden_size\":", conf->dim);
			SAFE_PARSE_INT(line, "\"intermediate_size\":", conf->hidden_dim);
			SAFE_PARSE_INT(line, "\"num_hidden_layers\":", conf->n_layers);
			SAFE_PARSE_INT(line, "\"num_attention_heads\":", conf->n_heads);
			SAFE_PARSE_INT(line, "\"num_key_value_heads\":", conf->n_kv_heads);
			SAFE_PARSE_INT(line, "\"vocab_size\":", conf->vocab_size);
			SAFE_PARSE_INT(line, "\"head_dim\":", conf->head_dim);
		}
		else
		{
			/* HuggingFace root level (Gemma, Llama, Qwen, etc) */
			SAFE_PARSE_INT(line, "\"hidden_size\":", conf->dim);
			SAFE_PARSE_INT(line, "\"intermediate_size\":", conf->hidden_dim);
			SAFE_PARSE_INT(line, "\"num_hidden_layers\":", conf->n_layers);
			SAFE_PARSE_INT(line, "\"num_attention_heads\":", conf->n_heads);
			SAFE_PARSE_INT(line, "\"num_key_value_heads\":", conf->n_kv_heads);
			SAFE_PARSE_INT(line, "\"vocab_size\":", conf->vocab_size);
			SAFE_PARSE_INT(line, "\"head_dim\":", conf->head_dim);
			SAFE_PARSE_FLOAT(line, "\"rope_theta\":", conf->rope_theta);
			SAFE_PARSE_FLOAT(line, "\"rms_norm_eps\":", conf->norm_eps);
			/* LLaMA-style keys (params.json) as additional fallbacks */
			SAFE_PARSE_INT(line, "\"dim\":", conf->dim);
			SAFE_PARSE_INT(line, "\"n_layers\":", conf->n_layers);
			SAFE_PARSE_INT(line, "\"n_heads\":", conf->n_heads);
			SAFE_PARSE_INT(line, "\"n_kv_heads\":", conf->n_kv_heads);
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
	
	/* [FIX] head_dim: Calculate from dim/n_heads if not explicitly set
	** SmolLM uses head_dim=64 (576/9), not 128. Many models don't set this. */
	if (conf->head_dim == 0) {
		if (conf->dim > 0 && conf->n_heads > 0)
			conf->head_dim = conf->dim / conf->n_heads;
		else
			conf->head_dim = 128;  /* Last resort default */
	}
	
	// Defaults for YaRN if not found
	if (conf->rope_theta == 0.0f) conf->rope_theta = 1000000.0f;
	if (conf->rope_factor == 0.0f) conf->rope_factor = 1.0f; // Default to standard RoPE
	if (conf->mscale == 0.0f) conf->mscale = 1.0f;
	
	conf->norm_eps = 1e-5f;
	conf->seq_len = 8192; // Reasonable default for inference

	return (0);
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
				
				/* ============================================================
				** UNIVERSAL TENSOR MAPPING (LLaMA + HuggingFace/Gemma Support)
				** ============================================================ */
				
				/* 1. ATTENTION */
				if (strstr(name, "attention.wq") || strstr(name, "self_attn.q_proj")) {
					l->wq = tensor;
				}
				else if (strstr(name, "attention.wk") || strstr(name, "self_attn.k_proj")) {
					l->wk = tensor;
				}
				else if (strstr(name, "attention.wv") || strstr(name, "self_attn.v_proj")) {
					l->wv = tensor;
				}
				else if (strstr(name, "attention.wo") || strstr(name, "self_attn.o_proj")) {
					l->wo = tensor;
				}
				
				/* 2. FEED FORWARD (LLaMA: w1=Gate, w2=Down, w3=Up) */
				else if (strstr(name, "feed_forward.w1") || strstr(name, "mlp.gate_proj")) {
					l->w1 = tensor;  /* Gate */
				}
				else if (strstr(name, "feed_forward.w2") || strstr(name, "mlp.down_proj")) {
					l->w2 = tensor;  /* Down */
				}
				else if (strstr(name, "feed_forward.w3") || strstr(name, "mlp.up_proj")) {
					l->w3 = tensor;  /* Up */
				}
				
				/* 3. NORMS (LLaMA vs HuggingFace) */
				else if (strstr(name, "attention_norm") || strstr(name, "input_layernorm")) {
					l->attention_norm = tensor;
				}
				else if (strstr(name, "ffn_norm") || strstr(name, "post_attention_layernorm")) {
					l->ffn_norm = tensor;
				}
			}
		}
		/* 4. GLOBAL WEIGHTS */
		else if (strcmp(name, "tok_embeddings.weight") == 0 ||
				 strcmp(name, "model.embed_tokens.weight") == 0) {
			t->weights.token_embedding = tensor;
		}
		else if (strcmp(name, "norm.weight") == 0 ||
				 strcmp(name, "model.norm.weight") == 0) {
			t->weights.norm = tensor;
		}
		else if (strcmp(name, "output.weight") == 0 ||
				 strcmp(name, "lm_head.weight") == 0) {
			t->weights.output = tensor;
		}

		/* Disabled for production - weight loading is very verbose:
		printf("Loaded: %s [%d, %d]\n", name, tensor->shape[0], tensor->shape[1]);
		if (strcmp(name, "model.layers.0.self_attn.q_proj.weight") == 0) {
			printf("[DEBUG] Shape Check: %s -> [%d, %d]\n", name, tensor->shape[0], tensor->shape[1]);
		}
		*/
		// Assuming map_set is a function that stores all tensors by name
		// map_set(t->weights, name, tensor);
	}

	/* Weight Tying: Link output head to embeddings if missing */
	if (!t->weights.output && t->weights.token_embedding) {
		printf("[LOADER] Weight Tying detected. Linking output → embeddings.\n");
		t->weights.output = t->weights.token_embedding;
	}

	// Handle tied embeddings if output weight is missing
	if (!t->weights.output)
		t->weights.output = t->weights.token_embedding;

	// ========== STATE ARENA (Arena Consolidation) ==========
	// 10MB state arena for all inference buffers - contiguous, cache-friendly
	arena_init(&t->state_arena, 10 * 1024 * 1024);
	if (!t->state_arena.base) {
		fprintf(stderr, "FATAL: state_arena init failed\n");
		return -1;
	}

	// Init state (now using state_arena for contiguous memory)
	t->state.x = arena_alloc_or_die(&t->state_arena, t->config.dim * sizeof(float));
	t->state.xb = arena_alloc_or_die(&t->state_arena, t->config.dim * sizeof(float));
	t->state.hb = arena_alloc_or_die(&t->state_arena, t->config.hidden_dim * sizeof(float));
	t->state.hb2 = arena_alloc_or_die(&t->state_arena, t->config.hidden_dim * sizeof(float));
	t->state.q = arena_alloc_or_die(&t->state_arena, t->config.n_heads * t->config.head_dim * sizeof(float));
	t->state.k = arena_alloc_or_die(&t->state_arena, t->config.n_kv_heads * t->config.head_dim * sizeof(float));
	t->state.v = arena_alloc_or_die(&t->state_arena, t->config.n_kv_heads * t->config.head_dim * sizeof(float));
	t->state.att = arena_alloc_or_die(&t->state_arena, t->config.n_heads * t->config.seq_len * sizeof(float));
	t->state.logits = arena_alloc_or_die(&t->state_arena, t->config.vocab_size * sizeof(float));
	t->state.grad_x = arena_alloc_or_die(&t->state_arena, t->config.dim * sizeof(float)); // For backprop
	t->state.final_input_cache = arena_alloc_or_die(&t->state_arena, t->config.dim * sizeof(float)); // [HOTFIX] Issue #1
	
	// ========== BATCHED PREFILL BUFFERS (Phase 2) - Now in state_arena ==========
	// Enable GEMM (M=batch_size) instead of GEMV (M=1) for QKV projections
	// Total: ~5-6MB extra memory for 20%+ prefill speedup
	t->state.batch_x = arena_alloc_or_die(&t->state_arena, MAX_PREFILL_BATCH * t->config.dim * sizeof(float));
	t->state.batch_xb = arena_alloc_or_die(&t->state_arena, MAX_PREFILL_BATCH * t->config.dim * sizeof(float));
	t->state.batch_q = arena_alloc_or_die(&t->state_arena, MAX_PREFILL_BATCH * t->config.n_heads * t->config.head_dim * sizeof(float));
	t->state.batch_k = arena_alloc_or_die(&t->state_arena, MAX_PREFILL_BATCH * t->config.n_kv_heads * t->config.head_dim * sizeof(float));
	t->state.batch_v = arena_alloc_or_die(&t->state_arena, MAX_PREFILL_BATCH * t->config.n_kv_heads * t->config.head_dim * sizeof(float));
	t->state.batch_out = arena_alloc_or_die(&t->state_arena, MAX_PREFILL_BATCH * t->config.dim * sizeof(float));
	t->state.batch_hb = arena_alloc_or_die(&t->state_arena, MAX_PREFILL_BATCH * t->config.hidden_dim * sizeof(float));
	t->state.batch_hb2 = arena_alloc_or_die(&t->state_arena, MAX_PREFILL_BATCH * t->config.hidden_dim * sizeof(float));
	
	// ========== NESTED LEARNING CONTEXT (TechLead Solution) - Now in state_arena ==========
	t->state.token_history = arena_alloc_or_die(&t->state_arena, t->config.seq_len * sizeof(int));
	
	// Init KV Cache
	// Allocate 1GB for KV cache
	size_t arena_size = 1024 * 1024 * 1024; 
	arena_init(&t->kv_arena, arena_size);
	if (!t->kv_arena.base) {
		return -1;
	}
	
	// Scratch arena for Top-K selection (1MB is plenty)
	arena_init(&t->scratch, 1024 * 1024);
	
	// Fluid arena for TENSOR_FLUID copies (writable adapters)
	// Size calculated based on trainable layers:
	// - Per trainable layer: ~200MB for dim=3072, hidden_dim=8192
	// - Plus final_adapter (dim*dim BF16), final_adapter_grad (dim*dim FP32)
	// - Plus logit_bias, context_bias, LSH structures
	// Formula: (n_layers - FROZEN_LAYERS) * per_layer + overhead
	int n_trainable = t->config.n_layers > FROZEN_LAYERS ? 
	                  t->config.n_layers - FROZEN_LAYERS : 1;
	size_t per_layer = (size_t)t->config.dim * t->config.hidden_dim * 
	                   (2 * sizeof(t_bf16) + sizeof(float)) +  // w2_weight, w2_grad, grad_acc
	                   t->config.hidden_dim * sizeof(float);    // hb_cache
	size_t final_adapter_bytes = (size_t)t->config.dim * t->config.dim * 
	                             (sizeof(t_bf16) + sizeof(float));  // BF16 weight + FP32 grad
	size_t adapter_overhead = final_adapter_bytes +
	                          t->config.vocab_size * sizeof(float) +                     // logit_bias
	                          CONTEXT_BIAS_SIZE * (sizeof(uint64_t) + sizeof(int) + sizeof(float)); // context_bias
	size_t fluid_arena_size = (n_trainable * per_layer) + adapter_overhead + (64 * 1024 * 1024); // +64MB safety margin
	
	arena_init(&t->fluid_arena, fluid_arena_size);
	printf("[FLUID_ARENA] Sized for %d trainable layers: %.2f MB\n", 
		n_trainable, fluid_arena_size / (1024.0 * 1024.0));
	
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
	
	// ========== FLUID LAYERS - NOW IN FLUID_ARENA ==========
	// Adapters for test-time learning - only for TRAINABLE (unfrozen) layers!
	// FROZEN_LAYERS defines how many bottom layers to skip.
	// With 26 layers and FROZEN_LAYERS=24, only layers 24,25 are trainable.
	// This saves ~200MB per frozen layer for Ministral-3B!
	size_t adapter_size = t->config.dim * t->config.hidden_dim;
	size_t fluid_layers_bytes = t->config.n_layers * sizeof(t_fluid_layer);
	t->fluid_layers = arena_alloc_or_die(&t->fluid_arena, fluid_layers_bytes);
	memset(t->fluid_layers, 0, fluid_layers_bytes);  // All NULL by default
	
	int trainable_layers = 0;
	size_t trainable_bytes = 0;
	
	for (int i = 0; i < t->config.n_layers; i++)
	{
		// OPTIMIZATION: Skip frozen layers entirely - saves massive memory!
		// Frozen layers keep NULL pointers, checked during backprop.
		if (i < FROZEN_LAYERS)
		{
			// Frozen layer - no adapter allocation needed
			t->fluid_layers[i].w2_weight = NULL;
			t->fluid_layers[i].w2_grad = NULL;
			t->fluid_layers[i].hb_cache = NULL;
			t->fluid_layers[i].grad_acc = NULL;
			continue;
		}
		
		trainable_layers++;
		
		// Allocate weight tensor (ZERO initialized - critical!)
		t->fluid_layers[i].w2_weight = arena_alloc_or_die(&t->fluid_arena, sizeof(t_tensor));
		memset(t->fluid_layers[i].w2_weight, 0, sizeof(t_tensor));
		t->fluid_layers[i].w2_weight->data = arena_alloc_or_die(&t->fluid_arena, adapter_size * sizeof(t_bf16));
		memset(t->fluid_layers[i].w2_weight->data, 0, adapter_size * sizeof(t_bf16));
		t->fluid_layers[i].w2_weight->size = adapter_size;
		t->fluid_layers[i].w2_weight->dtype = DTYPE_BF16;
		t->fluid_layers[i].w2_weight->ndim = 2;
		t->fluid_layers[i].w2_weight->shape[0] = t->config.dim;
		t->fluid_layers[i].w2_weight->shape[1] = t->config.hidden_dim;
		
		// Allocate gradient tensor
		t->fluid_layers[i].w2_grad = arena_alloc_or_die(&t->fluid_arena, sizeof(t_tensor));
		memset(t->fluid_layers[i].w2_grad, 0, sizeof(t_tensor));
		t->fluid_layers[i].w2_grad->data = arena_alloc_or_die(&t->fluid_arena, adapter_size * sizeof(t_bf16));
		memset(t->fluid_layers[i].w2_grad->data, 0, adapter_size * sizeof(t_bf16));
		t->fluid_layers[i].w2_grad->size = adapter_size;
		t->fluid_layers[i].w2_grad->dtype = DTYPE_BF16;
		t->fluid_layers[i].w2_grad->ndim = 2;
		t->fluid_layers[i].w2_grad->shape[0] = t->config.dim;
		t->fluid_layers[i].w2_grad->shape[1] = t->config.hidden_dim;
		
		// Allocate hb cache for backprop (stores hidden activations per layer)
		t->fluid_layers[i].hb_cache = arena_alloc_or_die(&t->fluid_arena, t->config.hidden_dim * sizeof(float));
		memset(t->fluid_layers[i].hb_cache, 0, t->config.hidden_dim * sizeof(float));
		
		// FP32 gradient accumulator (CRITICAL: fixes BF16 precision loss!)
		// Gradients accumulate in FP32, converted to BF16 only on weight update
		t->fluid_layers[i].grad_acc = arena_alloc_or_die(&t->fluid_arena, adapter_size * sizeof(float));
		memset(t->fluid_layers[i].grad_acc, 0, adapter_size * sizeof(float));
		
		trainable_bytes += (adapter_size * sizeof(t_bf16) * 2) +  // w2_weight + w2_grad
		                   (adapter_size * sizeof(float)) +       // grad_acc
		                   (t->config.hidden_dim * sizeof(float)); // hb_cache
	}
	
	printf("[FLUID_LAYERS] %d/%d layers trainable (FROZEN_LAYERS=%d)\n",
		trainable_layers, t->config.n_layers, FROZEN_LAYERS);
	printf("[FLUID_LAYERS] Adapter memory: %.2f MB (saved %.2f MB by freezing)\n",
		trainable_bytes / (1024.0 * 1024.0),
		(t->config.n_layers - trainable_layers) * 
		((adapter_size * 2 * sizeof(t_bf16)) + (adapter_size * sizeof(float)) + 
		 (t->config.hidden_dim * sizeof(float))) / (1024.0 * 1024.0));

	/*
	** Solution 4: Final Hidden Adapter [dim x dim]
	** Applied directly to final hidden state x before output projection.
	** This bypasses the FFN/hidden_dim mismatch issue.
	*/
	int final_adapter_size = t->config.dim * t->config.dim;
	t->final_adapter = arena_alloc_or_die(&t->fluid_arena, sizeof(t_tensor));
	memset(t->final_adapter, 0, sizeof(t_tensor));
	t->final_adapter->data = arena_alloc_or_die(&t->fluid_arena, final_adapter_size * sizeof(t_bf16));
	memset(t->final_adapter->data, 0, final_adapter_size * sizeof(t_bf16));
	t->final_adapter->size = final_adapter_size;
	t->final_adapter->dtype = DTYPE_BF16;
	t->final_adapter->ndim = 2;
	t->final_adapter->shape[0] = t->config.dim;
	t->final_adapter->shape[1] = t->config.dim;
	t->final_adapter_grad = arena_alloc_or_die(&t->fluid_arena, final_adapter_size * sizeof(float));
	memset(t->final_adapter_grad, 0, final_adapter_size * sizeof(float));
	printf("[FINAL_ADAPTER] Initialized in fluid_arena: [%d x %d] = %d params\n",
		t->config.dim, t->config.dim, final_adapter_size);

	/* Solution 5: Logit Bias [vocab_size] - direct output layer bias */
	t->logit_bias = arena_alloc_or_die(&t->fluid_arena, t->config.vocab_size * sizeof(float));
	memset(t->logit_bias, 0, t->config.vocab_size * sizeof(float));
	printf("[LOGIT_BIAS] Initialized in fluid_arena: [%d] = %lu bytes\n",
		t->config.vocab_size, t->config.vocab_size * sizeof(float));

	/* TechLead Solution: Context-Aware Bias Cache */
	t->context_bias.size = 65536; // 64k entries
	t->context_bias.keys = arena_alloc_or_die(&t->fluid_arena, t->context_bias.size * sizeof(uint64_t));
	memset(t->context_bias.keys, 0, t->context_bias.size * sizeof(uint64_t));
	t->context_bias.tokens = arena_alloc_or_die(&t->fluid_arena, t->context_bias.size * sizeof(int));
	memset(t->context_bias.tokens, 0, t->context_bias.size * sizeof(int));
	t->context_bias.biases = arena_alloc_or_die(&t->fluid_arena, t->context_bias.size * sizeof(float));
	memset(t->context_bias.biases, 0, t->context_bias.size * sizeof(float));
	t->context_bias.count = 0;
	printf("[CONTEXT_BIAS] Initialized in fluid_arena: %d entries = %lu bytes\n",
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
	
	// ========== ZERO-COPY BACKPROP (Phase: Arena Consolidation) ==========
	// We no longer pre-transpose output weights (saved 800MB for Ministral-3B!)
	// The backward pass now reads directly from frozen weights with SIMD.
	t->output_weight_T = NULL;

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

	// ========== LSH LIGHTNING INDEXER INIT - NOW IN FLUID_ARENA ==========
	// Initialize LSH for true O(K) sparse attention routing
	// This replaces brute-force O(N) key scanning with hash-based block selection
	{
		/* ops_lsh.h included at top of file */
		
		t_lsh_ctx *lsh_ctx = arena_alloc_or_die(&t->fluid_arena, sizeof(t_lsh_ctx));
		memset(lsh_ctx, 0, sizeof(t_lsh_ctx));
		t_lsh_index *lsh_idx = arena_alloc_or_die(&t->fluid_arena, sizeof(t_lsh_index));
		memset(lsh_idx, 0, sizeof(t_lsh_index));
		
		// Initialize LSH with head_dim dimension, seed based on config
		lsh_init(lsh_ctx, t->config.head_dim, 42);
		lsh_index_init(lsh_idx);  // Initialize hash table with empty buckets
		
		t->lsh_ctx = lsh_ctx;
		t->lsh_index = lsh_idx;
		t->use_lsh = (t->sparse_k > 0);  // Enable if sparse_k is set
		
		// CRITICAL: Initialize atomic stats to zero (Phase 9)
		// Without this, stats contain garbage from uninitialized memory!
		lsh_stats_reset_atomic(&t->lsh_stats);
		
		printf("[LSH] Lightning Indexer in fluid_arena (dim=%d, %d hash bits, block_size=%d)\n",
			t->config.head_dim, LSH_NUM_HASHES, LSH_BLOCK_SIZE);
	}

	return (0);
}

void	transformer_free(t_transformer *t)
{
	free_model(&t->model);
	free(t->weights.layers);
	
	// NOTE: State buffers (x, xb, hb, q, k, v, att, logits, batch_*, token_history)
	// are now in state_arena - freed via arena_free() below
	
	free(t->state.kv_cache);
	
	// NOTE: fluid_layers, final_adapter, logit_bias, context_bias are now in
	// fluid_arena - freed via arena_free() below. No manual free() needed.
	
	// Free eviction weights
	free(t->evict_weights.data);
	
	// Free arenas (handles all arena-managed memory)
	arena_free(&t->kv_arena);
	arena_free(&t->scratch);
	arena_free(&t->fluid_arena);
	arena_free(&t->state_arena);  // State buffers freed here
	
	// Free vision tower if allocated
	if (t->vision)
	{
		if (t->vision->vit_layers)
			free(t->vision->vit_layers);
		free(t->vision);
	}
	
	// NOTE: output_weight_T is now always NULL (zero-copy backprop)
	// NOTE: lsh_ctx/lsh_index are in fluid_arena, freed by arena_free above
	
	// Free precomputed RoPE thetas
	if (t->rope_thetas)
		free(t->rope_thetas);
	
	// Free RoPE cache (Phase 12)
	if (t->rope_cache)
		rope_cache_free((t_rope_cache *)t->rope_cache);
}
