/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   kv_cache_internal.h                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef KV_CACHE_INTERNAL_H
# define KV_CACHE_INTERNAL_H

# include "kv_cache.h"

typedef struct s_score_index
{
	float	val;
	int		idx;
}	t_score_index;

typedef struct s_evict_ctx
{
	int				n;
	int				heads;
	int				dim;
	t_kv_cache		*cache;
	t_score_index	*si;
}	t_evict_ctx;

void	compute_evict_scores(t_evict_ctx *ctx, const t_tensor *q,
			const t_tensor *w);
void	compact_kv_cache(t_evict_ctx *ctx, int keep_k);

#endif
