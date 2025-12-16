/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_silu.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/07 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/08 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include <math.h>
#include <stdio.h>
#include <omp.h>

/*
** FUSED AVX2 SwiGLU kernel: out = SiLU(gate) * val
** Uses fast_expf_avx2 for ~21x speedup over scalar expf
** See ops_activation.c for implementation details (Phase 6)
*/

#include "ops_activation.h"

static void	silu_mul_f32(t_tensor *out, const t_tensor *gate,
				const t_tensor *val)
{
	float	*out_data;
	float	*gate_data;
	float	*val_data;
	size_t	size;

	out_data = (float *)out->data;
	gate_data = (float *)gate->data;
	val_data = (float *)val->data;
	size = out->size;
	/* Use fused AVX2 kernel (Phase 6: 21x speedup) */
	op_silu_mul_fused_f32(out_data, gate_data, val_data, size);
}

static void	silu_mul_bf16(t_tensor *out, const t_tensor *gate,
				const t_tensor *val)
{
	t_bf16	*out_data;
	t_bf16	*gate_data;
	t_bf16	*val_data;
	size_t	size;

	out_data = (t_bf16 *)out->data;
	gate_data = (t_bf16 *)gate->data;
	val_data = (t_bf16 *)val->data;
	size = out->size;
	/* Use fused AVX2 kernel (Phase 6: 21x speedup) */
	op_silu_mul_fused_bf16((uint16_t *)out_data, (const uint16_t *)gate_data,
		(const uint16_t *)val_data, size);
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
