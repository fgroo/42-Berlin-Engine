/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_rope.h                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/16 20:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/16 20:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_ROPE_H
# define OPS_ROPE_H

# include <math.h>
# include <stdint.h>
# include <stdlib.h>

/*
** ============================================================================
** PRECOMPUTED ROPE CACHE (Phase 12: Sin/Cos Tables)
** ============================================================================
** Eliminates 53K sinf/cosf calls per token during generation.
** Tables precomputed at model init, then just FMA during inference.
**
** Memory: max_seq_len * head_dim/2 * 2 * 4 bytes
**         8192 * 64 * 2 * 4 = 4MB (very worthwhile trade-off)
** ============================================================================
*/

typedef struct s_rope_cache
{
	float	*cos_table;		/* [max_seq_len * head_dim/2] - 32-byte aligned */
	float	*sin_table;		/* [max_seq_len * head_dim/2] - 32-byte aligned */
	int		head_dim;
	int		max_seq_len;
}	t_rope_cache;

/*
** Initialize RoPE cache with precomputed sin/cos tables.
** Call once during model_init, before any inference.
**
** @param head_dim:     Dimension per attention head (e.g., 128)
** @param max_seq_len:  Maximum sequence length (e.g., 8192)
** @param theta_base:   RoPE theta base (e.g., 1000000.0f)
** @return:             Allocated and populated cache, or NULL on error
*/
t_rope_cache	*rope_cache_init(int head_dim, int max_seq_len, float theta_base);

/*
** Free RoPE cache memory.
*/
void			rope_cache_free(t_rope_cache *cache);

/*
** Apply RoPE rotation using precomputed tables (HOT PATH).
** No sinf/cosf calls - just table lookups and FMA.
**
** @param q:      Query vector [head_dim] (modified in place)
** @param k:      Key vector [head_dim] (modified in place)
** @param pos:    Absolute position in sequence
** @param cache:  Precomputed sin/cos tables
*/
void			rope_apply_cached(float *restrict q, float *restrict k,
					int pos, const t_rope_cache *restrict cache);

/*
** Apply RoPE to a single vector (Q or K separately).
** Used when processing batched prefill where Q and K are in different buffers.
*/
void			rope_apply_single_cached(float *restrict vec, int pos,
					const t_rope_cache *restrict cache);

/*
** Apply RoPE to multiple heads at once (FAST PATH).
** Processes n_heads contiguous head vectors in one function call.
** Each head is [head_dim] floats, stored contiguously.
*/
void			rope_apply_multihead_cached(float *restrict data, int n_heads,
					int pos, const t_rope_cache *restrict cache);

#endif
