/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_internal.h                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_INTERNAL_H
# define OPS_INTERNAL_H

# include "ops.h"

typedef struct s_score_idx
{
	float	val;
	int		idx;
}	t_score_idx;

typedef struct s_mm_ctx
{
	int				m;
	int				k;
	int				n;
	const t_tensor	*a;
	const t_tensor	*b;
}	t_mm_ctx;

typedef struct s_norm_ctx
{
	int		row;
	int		dim;
	float	inv_rms;
}	t_norm_ctx;

typedef struct s_light_ctx
{
	int		heads;
	int		dim;
	int		keys;
	t_bf16	*q;
	t_bf16	*k;
}	t_light_ctx;

#endif
