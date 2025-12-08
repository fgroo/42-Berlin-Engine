/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tensor.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "tensor.h"
#include <string.h>

static void	init_tensor_shape(t_tensor *t, int shape[], int ndim)
{
	int	i;

	i = 0;
	while (i < ndim)
	{
		t->shape[i] = shape[i];
		i++;
	}
	while (i < 4)
	{
		t->shape[i] = 1;
		i++;
	}
}

static void	init_tensor_strides(t_tensor *t, int ndim)
{
	int	stride;
	int	i;

	stride = 1;
	i = ndim - 1;
	while (i >= 0)
	{
		t->stride[i] = stride;
		t->size *= t->shape[i];
		stride *= t->shape[i];
		i--;
	}
	i = ndim;
	while (i < 4)
	{
		t->stride[i] = 0;
		i++;
	}
}

t_tensor	tensor_view(void *data, int shape[], int ndim)
{
	t_tensor	t;

	t.data = (t_bf16 *)data;
	t.ndim = ndim;
	t.size = 1;
	init_tensor_shape(&t, shape, ndim);
	init_tensor_strides(&t, ndim);
	return (t);
}

t_bf16	float_to_bf16(float f)
{
	unsigned int	bits;
	unsigned int	rounding;

	memcpy(&bits, &f, sizeof(float));
	rounding = (bits >> 16) & 1;
	bits += 0x7FFF + rounding;
	return ((t_bf16)(bits >> 16));
}

float	bf16_to_float(t_bf16 b)
{
	unsigned int	bits;
	float			result;

	bits = (unsigned int)b << 16;
	memcpy(&result, &bits, sizeof(float));
	return (result);
}
