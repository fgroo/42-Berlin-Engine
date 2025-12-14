/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_attention.h                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 18:30:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 18:30:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_ATTENTION_H
# define OPS_ATTENTION_H

# include "ops_internal.h"
# include "memory/kv_cache.h"
# include "memory/paged.h"

/*
** ============================================================================
** ATTENTION OPERATIONS
** ============================================================================
** Flash Attention V1 style: Online Softmax with O(1) memory for scores
** Supports both dense (full context) and sparse (Top-K) modes
** ============================================================================
*/

typedef struct s_attention_params
{
	const float		*q;			/* Query vector [n_heads * head_dim] */
	t_kv_cache		*kv_cache;	/* Key-Value cache for this layer */
	float			*output;	/* Output buffer [n_heads * head_dim] */
	int				n_heads;	/* Number of query heads */
	int				n_kv_heads;	/* Number of KV heads (for GQA) */
	int				head_dim;	/* Dimension per head */
	float			scale;		/* 1/sqrt(head_dim) */
	int				*topk_idx;	/* Optional: indices for sparse attn */
	int				attend_k;	/* Number of keys to attend to */
}	t_attention_params;

/*
** Flash Attention: Single-pass attention with online softmax
** Memory: O(head_dim) instead of O(seq_len)
*/
void	flash_attention_head(float *out, const float *q,
			const t_bf16 *k_cache, const t_bf16 *v_cache,
			int seq_len, int head_dim, float scale);

/*
** Multi-head attention with optional sparse indices
** Handles GQA (Grouped Query Attention) head mapping
*/
void	op_multihead_attention(t_attention_params *p);

/*
** Dense attention (attends to all positions up to kv_len)
*/
void	op_attention_dense(t_attention_params *p, int kv_len);

/*
** Sparse attention (attends only to topk_idx positions)
*/
void	op_attention_sparse(t_attention_params *p);

/*
** ============================================================================
** BLOCK-BASED PAGED ATTENTION (for sparse attention foundation)
** ============================================================================
*/

/*
** Multi-head paged attention (iterates over block page table)
*/
void	op_paged_attention(float *output, const float *q,
			t_paged_kv_cache *pkv, int n_heads, int n_kv_heads,
			int head_dim, float scale);

/*
** Multi-head paged sparse attention (only selected blocks)
*/
void	op_paged_attention_sparse(float *output, const float *q,
			t_block_manager *bm, const int *selected_blocks, int n_selected,
			int n_heads, int n_kv_heads, int head_dim, float scale);

#endif
