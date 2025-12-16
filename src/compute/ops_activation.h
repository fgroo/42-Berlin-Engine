/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_activation.h                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 23:15:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/15 23:15:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef OPS_ACTIVATION_H
# define OPS_ACTIVATION_H

# include <stddef.h>
# include <stdint.h>

/*
** Fused SiLU-Mul AVX2 Kernels (Phase 6)
** out = SiLU(gate) * up = (gate * sigmoid(gate)) * up
*/

/* F32 version - single-threaded SIMD */
void	op_silu_mul_fused_f32(float *dst, const float *gate,
							const float *up, size_t n);

/* F32 version - OpenMP parallelized for large n */
void	op_silu_mul_fused_f32_omp(float *dst, const float *gate,
								const float *up, size_t n);

/* BF16 version - load BF16, compute F32, store BF16 */
void	op_silu_mul_fused_bf16(uint16_t *dst, const uint16_t *gate,
							const uint16_t *up, size_t n);

#endif
