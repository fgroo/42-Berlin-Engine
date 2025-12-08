/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   kv_cache.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "kv_cache.h"
#include "compute/ops.h"
#include <string.h>
#include <stdio.h>

void	kv_cache_init(t_kv_cache *c, t_arena *a, t_kv_init_params *p)
{
	int		shape[3];
	size_t	size_bytes;

	c->max_seq_len = p->max_seq;
	c->num_heads = p->num_heads;
	c->head_dim = p->head_dim;
	c->current_seq_len = 0;
	shape[0] = p->max_seq;
	shape[1] = p->num_heads;
	shape[2] = p->head_dim;
	size_bytes = p->max_seq * p->num_heads * p->head_dim * sizeof(t_bf16);
	c->k = tensor_view(arena_alloc(a, size_bytes), shape, 3);
	c->v = tensor_view(arena_alloc(a, size_bytes), shape, 3);
}

static void	convert_f32_to_bf16(t_bf16 *dest, float *src, int n)
{
	int	i;

	i = 0;
	while (i < n)
	{
		dest[i] = float_to_bf16(src[i]);
		i++;
	}
}

static void	copy_kv_data(t_bf16 *dest, const t_tensor *src, int block_size)
{
	if (src->dtype == DTYPE_F32)
		convert_f32_to_bf16(dest, (float *)src->data, block_size);
	else
		memcpy(dest, src->data, block_size * sizeof(t_bf16));
}

void	kv_cache_append(t_kv_cache *c, const t_tensor *k_in,
			const t_tensor *v_in)
{
	int		idx;
	int		block_size;
	t_bf16	*dest_k;
	t_bf16	*dest_v;

	if (c->current_seq_len >= c->max_seq_len)
	{
		fprintf(stderr, "KV Cache Full!\n");
		return ;
	}
	idx = c->current_seq_len;
	block_size = c->num_heads * c->head_dim;
	dest_k = (t_bf16 *)c->k.data + idx * block_size;
	dest_v = (t_bf16 *)c->v.data + idx * block_size;
	copy_kv_data(dest_k, k_in, block_size);
	copy_kv_data(dest_v, v_in, block_size);
	c->current_seq_len++;
}
