/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   gemm.h                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 21:20:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 21:20:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef GEMM_H
# define GEMM_H

# include <stdint.h>

/*
** Tiled GEMM with L2 cache blocking
** C[M x N] = A[M x K] * B[K x N]
** BF16 inputs, F32 accumulation
**
** @param zero_c: If non-zero, zero C before computation
*/
void	ops_gemm_bf16_tiled(
			int M, int N, int K,
			const uint16_t *A, int lda,
			const uint16_t *B, int ldb,
			float *C, int ldc,
			int zero_c);

/*
** Naive scalar GEMM for comparison
*/
void	ops_gemm_bf16_naive(
			int M, int N, int K,
			const uint16_t *A, int lda,
			const uint16_t *B, int ldb,
			float *C, int ldc);

#endif
