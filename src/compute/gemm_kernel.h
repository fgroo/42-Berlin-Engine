/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   gemm_kernel.h                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 16:00:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/15 16:00:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef GEMM_KERNEL_H
# define GEMM_KERNEL_H

# include <stdint.h>

/*
** 6x16 Register-Blocked GEMM Microkernel
** Computes C[6x16] += A[6xK] * B[Kx16] with BF16 inputs and F32 accumulation
*/
void	gemm_microkernel_6x16_bf16(
			int K,
			const uint16_t *A, int lda,
			const uint16_t *B, int ldb,
			float *C, int ldc);

/*
** Naive scalar GEMM for comparison
*/
void	gemm_naive_bf16(
			int M, int N, int K,
			const uint16_t *A, int lda,
			const uint16_t *B, int ldb,
			float *C, int ldc);

#endif
