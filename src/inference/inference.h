#ifndef INFERENCE_H
# define INFERENCE_H

# include "tensor/tensor.h"
# include "loader/loader.h"
# include "tokenizer/tokenizer.h"
# include "memory/kv_cache.h"
# include "memory/arena.h"

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
	t_kv_cache	*kv_cache; // Array of KV caches [n_layers]
}	t_inference_state;

// =============== NESTED LEARNING ===============
// Fluid weights = small adapters that learn during inference
typedef struct s_fluid_layer
{
	t_tensor	*w2_weight;  // Adapter weight [dim x hidden_dim] (BF16)
	t_tensor	*w2_grad;    // Adapter gradient [dim x hidden_dim] (BF16)
	float		*hb_cache;   // Cached hidden activations [hidden_dim] for backprop
}	t_fluid_layer;

typedef struct s_transformer
{
	t_transformer_config	config;
	t_transformer_weights	weights;
	t_inference_state		state;
	t_model					model;
	t_arena					kv_arena;
	t_arena					scratch;
	int						sparse_k;
	int						nested_learning;
	float					nested_lr;
	t_fluid_layer			*fluid_layers;
	int						evict_threshold;
	int						evict_keep_k;
	t_tensor				evict_weights;
	t_bf16					*output_weight_T;
}	t_transformer;

int		transformer_init(t_transformer *t, const char *model_path, const char *config_path);
void	transformer_free(t_transformer *t);
float	*transformer_forward(t_transformer *t, int token, int pos);
void	transformer_backward_step(t_transformer *t, int target_token, int pos);

#endif
