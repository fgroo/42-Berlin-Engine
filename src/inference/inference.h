#ifndef INFERENCE_H
# define INFERENCE_H

# include "tensor/tensor.h"
# include "loader/loader.h"
# include "tokenizer/tokenizer.h"
# include "memory/kv_cache.h"
# include "memory/paged.h"
# include "memory/arena.h"
# include "core/types.h"  /* MOPD: t_sparse_prob, t_distill_request */
# include <stdatomic.h>  // C11 atomics for thread-safe NL counters
# include "nested/nl_counters.h"  // CAS-based lock-free NL counters
# include "compute/ops_lsh.h"  // Thread-safe LSH stats (Phase 9)

typedef struct s_transformer_config
{
	int		dim; // hidden_size (3072)
	int		hidden_dim; // intermediate_size (9216)
	int		n_layers; // num_hidden_layers (26)
	int		n_heads; // num_attention_heads (32)
	int		n_kv_heads; // num_key_value_heads (8)
	int		vocab_size; // 131072
	int		seq_len; // max_position_embeddings (262144 or smaller for inference)
	int		head_dim; 	// RoPE / YaRN
	float	rope_theta;
	float	rope_factor;
	float	beta_fast;
	float	beta_slow;
	float	mscale;
	int		orig_ctx_len; // 1000000.0
	float	norm_eps; // 1e-5
}	t_transformer_config;

typedef struct s_layer_weights
{
	t_tensor	*wq;
	t_tensor	*wk;
	t_tensor	*wv;
	t_tensor	*wo;
	t_tensor	*w1; // gate
	t_tensor	*w2; // down
	t_tensor	*w3; // up
	t_bf16		*w_fused_gate_up;  // [hidden_dim*2, dim] fused w1|w3 (Phase 11)
	t_tensor	*attention_norm;
	t_tensor	*ffn_norm;
}	t_layer_weights;

typedef struct s_transformer_weights
{
	t_tensor		*token_embedding;
	t_tensor		*norm;
	t_tensor		*output; // Usually tied to token_embedding
	t_layer_weights	*layers;
}	t_transformer_weights;

typedef struct s_inference_state
{
	float	*x; // Current state [dim]
	float	*xb; // Buffer [dim]
	float	*hb; // Buffer [hidden_dim]
	float	*hb2; // Buffer [hidden_dim] (for MLP up_proj)
	float	*q; // [dim]
	float	*k; // [n_kv_heads * head_dim]
	float	*v; // [n_kv_heads * head_dim]
	float	*att; // [n_heads * seq_len] (Attention scores)
	float	*logits; // [vocab_size]
	float	*grad_x; // Gradient buffer for backprop [dim]
	// [HOTFIX] Cache for final layer input during backprop
	// Replaces dangerous static buffer in forward pass
	float	*final_input_cache; // [dim]
	t_kv_cache	*kv_cache; // Array of KV caches [n_layers]
	// ========== BATCHED PREFILL BUFFERS (Phase 2) ==========
	// These enable GEMM (M>1) instead of GEMV (M=1) for QKV projections
	float	*batch_x;    // [batch_size x dim] - batched embeddings
	float	*batch_xb;   // [batch_size x dim] - batched normalized
	float	*batch_q;    // [batch_size x n_heads * head_dim]
	float	*batch_k;    // [batch_size x n_kv_heads * head_dim]
	float	*batch_v;    // [batch_size x n_kv_heads * head_dim]
	float	*batch_out;  // [batch_size x dim] - attention output
	float	*batch_hb;   // [batch_size x hidden_dim] - FFN buffer
	float	*batch_hb2;  // [batch_size x hidden_dim] - FFN buffer 2
	// ========== FFN FUSION (Phase 11 v2) ==========
	float	*hb_fused;   // [hidden_dim * 2] - fused gate+up output
	// ========== NESTED LEARNING CONTEXT (TechLead Solution) ==========
	int		*token_history; // [seq_len] - stores input tokens for backward pass
}	t_inference_state;

// Maximum batch size for prefill (tune for L2 cache: 64 * 3072 * 4 = 768KB)
#define MAX_PREFILL_BATCH 64

// =============== NESTED LEARNING ===============
// Fluid weights = small adapters that learn during inference
// MIXED PRECISION: Weights in BF16, gradients accumulate in FP32
typedef struct s_fluid_layer
{
	t_tensor	*w2_weight;  // Adapter weight [dim x hidden_dim] (BF16)
	t_tensor	*w2_grad;    // Adapter gradient [dim x hidden_dim] (BF16) - DEPRECATED
	float		*grad_acc;   // FP32 gradient accumulator [dim x hidden_dim] - CRITICAL!
	float		*hb_cache;   // Cached hidden activations [hidden_dim] for backprop
}	t_fluid_layer;

// =============== VISION TOWER (LAZY LOADING) ===============
// Pointer is NULL in text-only mode - weights stay in mmap, never paged
// Set via activate_vision_tower() to enable multimodal inference
typedef struct s_vision_tower
{
	t_tensor	*patch_embed;      // Patch embedding [patch_size² * 3, embed_dim]
	t_tensor	*pos_embed;        // Position embedding
	t_tensor	**vit_layers;      // Pre-allocated pointer array to ViT blocks
	t_tensor	*projector;        // Vision -> Text projection
	int			num_vit_layers;
	int			enabled;           // 0 = lazy (not paged), 1 = active
}	t_vision_tower;

typedef struct s_transformer
{
	t_transformer_config	config;
	t_transformer_weights	weights;
	t_inference_state		state;
	t_model					model;
	t_arena					kv_arena;
	t_arena					scratch;
	t_arena					fluid_arena;   // Arena for FLUID tensor copies (RW)
	int						sparse_k;
	int						nested_learning;
	float					nested_lr;
	t_fluid_layer			*fluid_layers;
	int						evict_threshold;
	int						evict_keep_k;
	t_tensor				evict_weights;
	t_bf16					*output_weight_T;
	float				*rope_thetas;      // Precomputed RoPE theta values [head_dim/2]
	void				*rope_cache;       // Precomputed sin/cos tables (t_rope_cache*) [Phase 12]
	void				*lsh_ctx;          // LSH context for sparse routing (t_lsh_ctx*)
	void				*lsh_index;        // LSH block index (t_lsh_index*)
	t_lsh_stats_atomic	lsh_stats;     // Thread-safe LSH diagnostics (Phase 9)
	int					use_lsh;           // Enable LSH-based sparse attention
	t_vision_tower			*vision;       // NULL = text-only mode (lazy vision)
	/*
	** Nested Learning Counters - LOCK-FREE CAS (Phase 2)
	** Step + Skipped are packed into a single 64-bit atomic to ensure
	** consistent reads (no torn reads from interleaved updates).
	** See nl_counters.h for the CAS-based update API.
	*/
	t_nl_atomic_state		nl_state;          // Packed step+skipped + actual_steps
	int		persistent_mode; // 1 = retain fluid weights across turns
	int		raw_mode;        // 1 = no chat template (raw completion)
	// Paged KV Cache for Sparse Attention (O(K) memory access)
	t_block_manager		block_manager;     // Block pool for all layers
	t_paged_kv_cache	*paged_kv;         // Per-layer paged cache views
	int					use_paged_kv;      // 0 = old linear cache, 1 = paged blocks
	/*
	** Final Hidden Adapter [dim x dim] - Solution 4
	** Applied directly to final hidden state x before output projection.
	** Trained via backward pass with gradient accumulation.
	*/
	t_tensor			*final_adapter;      // [dim x dim] BF16 weights
	float				*final_adapter_grad; // [dim x dim] FP32 gradient accumulator
	/*
	** Logit Bias [vocab_size] - Solution 5 (Global)
	*/
	float				*logit_bias;         // [vocab_size] FP32 bias
	
	/*
	** Context-Aware Bias Cache - TechLead Solution
	** Maps (pos, prev_token) -> (target_token, bias)
	*/
	struct {
		uint64_t	*keys;      // (pos << 32) | prev_token
		int			*tokens;    // target_token_id
		float		*biases;    // bias value
		int			size;       // hash table size
		int			count;      // current number of entries
	} context_bias;
}	t_transformer;

int		transformer_init(t_transformer *t, const char *model_path, const char *config_path);
void	transformer_free(t_transformer *t);
float	*transformer_forward(t_transformer *t, int token, int pos);
void	transformer_backward_step(t_transformer *t, int target_token, int pos);

/*
** Batched prefill: Process multiple tokens using GEMM instead of GEMV
** Significantly faster for long prompts (20%+ speedup expected)
** @param t: Transformer struct
** @param tokens: Array of token IDs  
** @param batch_size: Number of tokens to process (max MAX_PREFILL_BATCH)
** @param start_pos: Starting position in sequence
*/
void	forward_prefill_batch(t_transformer *t, const int *tokens,
			int batch_size, int start_pos);

// Vision tower control
int		activate_vision_tower(t_transformer *t);
void	deactivate_vision_tower(t_transformer *t);

// Backward pass (nested/backward.c) - FP32 gradient accumulation
void	backward_zero_grads(t_transformer *t);
void	backward_apply_grads(t_transformer *t, float lr);

/*
** ============================================================================
** MOPD (Multi-Teacher On-Policy Distillation) - Phase 1
** ============================================================================
** Splits backward pass into modular components for flexible loss functions.
** ============================================================================
*/

/*
** Core backprop mechanism: Takes d_logits and propagates through all layers.
** This is the "heavy lifter" - does MatMul transpose, accumulates grads.
** @param d_logits  Gradient vector [vocab_size], dL/dlogits
** @param pos       Current sequence position
*/
void	backward_propagate_logits(t_transformer *t, float *d_logits, int pos);

/*
** Cross-Entropy backward (standard training).
** Computes gradient as: softmax(logits) - one_hot(target)
*/
void	backward_step_ce(t_transformer *t, int target_token, int pos);

/*
** MOPD Distillation backward pass.
** Computes gradient as: softmax(logits) - mixed_target
** where mixed_target = alpha * teacher + (1-alpha) * one_hot
**
** @param teacher_probs  Sparse probability distribution from teacher
** @param num_probs      Number of entries in teacher_probs
** @param target_token   Ground truth token (for hard label component)
** @param alpha          Mixing coefficient [0.0 = pure CE, 1.0 = pure teacher]
*/
void	backward_step_distill(t_transformer *t, t_sparse_prob *teacher_probs,
			int num_probs, int target_token, float alpha, int pos);

#endif
