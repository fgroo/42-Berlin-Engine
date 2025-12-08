/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_silu.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/07 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/08 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include <math.h>
#include <stdio.h>
#include <omp.h>

/*
** OpenMP-parallelized SwiGLU kernel: out = SiLU(gate) * val
** SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
** 
** Note: exp() is hard to vectorize without external libs.
** OpenMP parallelization across hidden_dim (9216) gives good speedup.
** Each of 8 threads processes ~1152 elements.
*/

static void	silu_mul_f32(t_tensor *out, const t_tensor *gate,
				const t_tensor *val)
{
	size_t	size;
	float	*out_data;
	float	*gate_data;
	float	*val_data;

	out_data = (float *)out->data;
	gate_data = (float *)gate->data;
	val_data = (float *)val->data;
	size = out->size;
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < size; i++)
	{
		float g = gate_data[i];
		float v = val_data[i];
		float sigmoid = 1.0f / (1.0f + expf(-g));
		out_data[i] = g * sigmoid * v;
	}
}

static void	silu_mul_bf16(t_tensor *out, const t_tensor *gate,
				const t_tensor *val)
{
	size_t	size;
	t_bf16	*out_data;
	t_bf16	*gate_data;
	t_bf16	*val_data;

	out_data = (t_bf16 *)out->data;
	gate_data = (t_bf16 *)gate->data;
	val_data = (t_bf16 *)val->data;
	size = out->size;
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < size; i++)
	{
		float g = bf16_to_float(gate_data[i]);
		float v = bf16_to_float(val_data[i]);
		float sigmoid = 1.0f / (1.0f + expf(-g));
		out_data[i] = float_to_bf16(g * sigmoid * v);
	}
}

static int	check_silu_dtypes(t_tensor *out, const t_tensor *gate,
				const t_tensor *val)
{
	if (out->dtype == DTYPE_F32 && gate->dtype == DTYPE_F32)
		return (val->dtype == DTYPE_F32);
	return (0);
}

void	op_silu_mul(t_tensor *out, const t_tensor *gate, const t_tensor *val)
{
	int	all_bf16;

	if (gate->size != out->size || val->size != out->size)
		return ;
	if (check_silu_dtypes(out, gate, val))
	{
		silu_mul_f32(out, gate, val);
		return ;
	}
	all_bf16 = (out->dtype == DTYPE_BF16);
	all_bf16 = all_bf16 && (gate->dtype == DTYPE_BF16);
	all_bf16 = all_bf16 && (val->dtype == DTYPE_BF16);
	if (all_bf16)
		silu_mul_bf16(out, gate, val);
	else
		fprintf(stderr, "SiLU unsupported dtypes\n");
}
