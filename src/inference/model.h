/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   model.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/22 19:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/22 19:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef MODEL_H
# define MODEL_H

# include "inference.h"

/*
** ============================================================================
** MODEL LOADING API
** ============================================================================
** High-level functions for loading transformer models from safetensors files.
** Supports both LLaMA-style and HuggingFace-style tensor naming conventions.
*/

/*
** Initialize transformer from model files.
** @param t         Transformer struct to initialize
** @param model_path Path to .safetensors weights file
** @param config_path Path to config.json file
** @return 0 on success, -1 on error
*/
int		transformer_init(t_transformer *t, const char *model_path,
			const char *config_path);

/*
** Free all resources associated with a transformer.
** @param t Transformer to free
*/
void	transformer_free(t_transformer *t);

/*
** ============================================================================
** CONFIGURATION PARSING
** ============================================================================
** Supports both LLaMA-style (dim, n_layers) and HuggingFace-style
** (hidden_size, num_hidden_layers) configuration keys.
**
** VLM models (Ministral) use nested "text_config" block.
** Standard models (Gemma, SmolLM) use root-level keys.
*/

/*
** ============================================================================
** UNIVERSAL WEIGHT MAPPER
** ============================================================================
** Automatically maps tensor names from different frameworks:
**
** Attention:
**   LLaMA:      attention.wq, attention.wk, attention.wv, attention.wo
**   HuggingFace: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj
**
** Feed Forward:
**   LLaMA:      feed_forward.w1 (gate), w2 (down), w3 (up)
**   HuggingFace: mlp.gate_proj, mlp.down_proj, mlp.up_proj
**
** Norms:
**   LLaMA:      attention_norm, ffn_norm
**   HuggingFace: input_layernorm, post_attention_layernorm
**
** Embeddings:
**   LLaMA:      tok_embeddings.weight, output.weight
**   HuggingFace: model.embed_tokens.weight, lm_head.weight
*/

#endif /* MODEL_H */
