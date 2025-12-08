/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid_backward.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fluid.h"
#include "compute/ops.h"
#include <stdio.h>

static void	backward_inner(t_tensor *gw, const t_tensor *x,
				const t_tensor *grad_out, int dims[3])
{
	int		i;
	int		j;
	int		b;
	float	sum;
	t_bf16	*x_data;
	t_bf16	*go_data;
	t_bf16	*gw_data;

	x_data = (t_bf16 *)x->data;
	go_data = (t_bf16 *)grad_out->data;
	gw_data = (t_bf16 *)gw->data;
	i = 0;
	while (i < dims[1])
	{
		j = 0;
		while (j < dims[2])
		{
			sum = 0.0f;
			b = 0;
			while (b < dims[0])
			{
				sum += bf16_to_float(x_data[b * dims[1] + i])
					* bf16_to_float(go_data[b * dims[2] + j]);
				b++;
			}
			gw_data[i * dims[2] + j] = float_to_bf16(sum);
			j++;
		}
		i++;
	}
}

void	backward_linear(t_fluid_param *param, const t_tensor *x,
			const t_tensor *grad_output)
{
	int			dims[3];
	t_tensor	*gw;

	dims[0] = x->shape[0];
	dims[1] = x->shape[1];
	dims[2] = grad_output->shape[1];
	gw = param->grad;
	if (gw->shape[0] != dims[1] || gw->shape[1] != dims[2])
	{
		fprintf(stderr, "Backward dim mismatch\n");
		return ;
	}
	backward_inner(gw, x, grad_output, dims);
}
