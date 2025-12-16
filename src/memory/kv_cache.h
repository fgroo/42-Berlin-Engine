/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   kv_cache.h                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef KV_CACHE_H
# define KV_CACHE_H

# include "arena.h"
# include "tensor/tensor.h"

typedef struct s_kv_cache
{
	t_tensor	k;
	t_tensor	v;
	int			max_seq_len;
	int			num_heads;
	int			head_dim;
	int			current_seq_len;
}	t_kv_cache;

typedef struct s_kv_init_params
{
	int	max_seq;
	int	num_heads;
	int	head_dim;
}	t_kv_init_params;

typedef struct s_evict_params
{
	const t_tensor	*q;
	const t_tensor	*w;
	int				keep_k;
	t_arena			*scratch;
}	t_evict_params;

void	kv_cache_init(t_kv_cache *c, t_arena *a, t_kv_init_params *p);
void	kv_cache_append(t_kv_cache *c, const t_tensor *k, const t_tensor *v);
void	kv_cache_evict(t_kv_cache *c, t_evict_params *p);

#endif
