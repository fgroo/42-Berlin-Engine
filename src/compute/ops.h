/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops.h                                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_H
# define OPS_H

# include "tensor/tensor.h"
# include "memory/arena.h"

typedef struct s_rope_ctx
{
	int		head_dim;
	int		pos;
	float	theta_base;
	// YaRN parameters
	float	beta_fast;
	float	beta_slow;
	float	factor;
	float	mscale;
	int		orig_ctx;
	float	*thetas_cache;  // Optional: precomputed thetas [head_dim/2], NULL = compute on the fly
}	t_rope_ctx;

int		op_matmul(t_tensor *out, const t_tensor *a, const t_tensor *b);
void	op_rmsnorm(t_tensor *out, const t_tensor *x,
			const t_tensor *w, float epsilon);
void	op_rope(t_tensor *x, int pos, const t_rope_ctx *ctx);
void	op_lightning_score(t_tensor *scores, const t_tensor *q,
			const t_tensor *k, const t_tensor *w);
void	op_topk_select(int *indices, const t_tensor *scores,
			int k, t_arena *scratch);
void	op_silu_mul(t_tensor *out, const t_tensor *gate, const t_tensor *val);

#endif
