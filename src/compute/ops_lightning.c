/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_lightning.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include <stdio.h>

static float	dot_relu(t_bf16 *q, t_bf16 *k, int dim)
{
	float	dot;
	int		d;

	dot = 0.0f;
	d = 0;
	while (d < dim)
	{
		dot += bf16_to_float(q[d]) * bf16_to_float(k[d]);
		d++;
	}
	if (dot < 0)
		dot = 0.0f;
	return (dot);
}

static void	score_key(t_tensor *sc, t_light_ctx *c, int s, const t_tensor *w)
{
	int		h;
	float	sum;
	t_bf16	*w_data;
	t_bf16	*sc_data;

	w_data = (t_bf16 *)w->data;
	sc_data = (t_bf16 *)sc->data;
	sum = 0.0f;
	h = 0;
	while (h < c->heads)
	{
		sum += bf16_to_float(w_data[h])
			* dot_relu(c->q + h * c->dim, c->k + s * c->dim, c->dim);
		h++;
	}
	sc_data[s] = float_to_bf16(sum);
}

void	op_lightning_score(t_tensor *scores, const t_tensor *q,
			const t_tensor *k, const t_tensor *w)
{
	t_light_ctx	c;
	int			s;

	c.heads = q->shape[0];
	c.dim = q->shape[1];
	c.keys = k->shape[0];
	c.q = q->data;
	c.k = k->data;
	if (k->shape[1] != c.dim)
	{
		fprintf(stderr, "Lightning dim mismatch\n");
		return ;
	}
	s = 0;
	while (s < c.keys)
	{
		score_key(scores, &c, s, w);
		s++;
	}
}
