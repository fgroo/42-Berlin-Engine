/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   gemm_kernel.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 16:00:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 16:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** 6x16 REGISTER-BLOCKED GEMM MICROKERNEL (Phase 4)
** ===========================================================================
** C[6x16] += A[6xK] * B[Kx16]   (BF16 -> F32 accumulation)
**
** Register allocation (16 YMM registers on AVX2):
** - 12 registers for C accumulators (6 rows x 2 chunks of 8)
** - 2 registers for B columns (loaded once per K iteration)
** - 2 registers for A broadcast + scratch
**
** This is the theoretical maximum utilization for AVX2 without spilling.
** ===========================================================================
*/

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include "ops_simd.h"

/*
** 6x16 Microkernel: BF16 inputs, F32 accumulation
**
** @param K:   Inner dimension (dot product length)
** @param A:   Row-major matrix [M x K] as BF16, we use rows 0-5
** @param lda: Leading dimension of A (stride between rows)
** @param B:   Row-major matrix [K x N] as BF16, we use cols 0-15
** @param ldb: Leading dimension of B (stride between rows = N)
** @param C:   Output matrix [M x N] as F32
** @param ldc: Leading dimension of C
*/
void	gemm_microkernel_6x16_bf16(
	int K,
	const uint16_t *A, int lda,
	const uint16_t *B, int ldb,
	float *C, int ldc)
{
	/* 12 Accumulator registers: c[row][chunk] */
	__m256	c00, c01;  /* Row 0, cols 0-7 and 8-15 */
	__m256	c10, c11;  /* Row 1 */
	__m256	c20, c21;  /* Row 2 */
	__m256	c30, c31;  /* Row 3 */
	__m256	c40, c41;  /* Row 4 */
	__m256	c50, c51;  /* Row 5 */
	__m256	vb0, vb1;  /* B columns 0-7 and 8-15 */
	__m256	va;        /* Broadcasted A value */
	__m128i	b_bf16_0, b_bf16_1;
	int		k;

	/* Load existing C values for accumulation (critical for K-tiling) */
	c00 = _mm256_loadu_ps(&C[0 * ldc + 0]);
	c01 = _mm256_loadu_ps(&C[0 * ldc + 8]);
	c10 = _mm256_loadu_ps(&C[1 * ldc + 0]);
	c11 = _mm256_loadu_ps(&C[1 * ldc + 8]);
	c20 = _mm256_loadu_ps(&C[2 * ldc + 0]);
	c21 = _mm256_loadu_ps(&C[2 * ldc + 8]);
	c30 = _mm256_loadu_ps(&C[3 * ldc + 0]);
	c31 = _mm256_loadu_ps(&C[3 * ldc + 8]);
	c40 = _mm256_loadu_ps(&C[4 * ldc + 0]);
	c41 = _mm256_loadu_ps(&C[4 * ldc + 8]);
	c50 = _mm256_loadu_ps(&C[5 * ldc + 0]);
	c51 = _mm256_loadu_ps(&C[5 * ldc + 8]);

	/* K-loop: dot product dimension */
	k = 0;
	while (k < K)
	{
		/* Load B row k, cols 0-15 (16 BF16 values -> 2x8 F32) */
		b_bf16_0 = _mm_loadu_si128((const __m128i *)&B[k * ldb + 0]);
		b_bf16_1 = _mm_loadu_si128((const __m128i *)&B[k * ldb + 8]);
		vb0 = mm256_cvtbf16_ps(b_bf16_0);
		vb1 = mm256_cvtbf16_ps(b_bf16_1);

		/* Row 0: broadcast A[0,k], FMA with B */
		va = mm256_broadcast_bf16(A[0 * lda + k]);
		c00 = _mm256_fmadd_ps(va, vb0, c00);
		c01 = _mm256_fmadd_ps(va, vb1, c01);

		/* Row 1 */
		va = mm256_broadcast_bf16(A[1 * lda + k]);
		c10 = _mm256_fmadd_ps(va, vb0, c10);
		c11 = _mm256_fmadd_ps(va, vb1, c11);

		/* Row 2 */
		va = mm256_broadcast_bf16(A[2 * lda + k]);
		c20 = _mm256_fmadd_ps(va, vb0, c20);
		c21 = _mm256_fmadd_ps(va, vb1, c21);

		/* Row 3 */
		va = mm256_broadcast_bf16(A[3 * lda + k]);
		c30 = _mm256_fmadd_ps(va, vb0, c30);
		c31 = _mm256_fmadd_ps(va, vb1, c31);

		/* Row 4 */
		va = mm256_broadcast_bf16(A[4 * lda + k]);
		c40 = _mm256_fmadd_ps(va, vb0, c40);
		c41 = _mm256_fmadd_ps(va, vb1, c41);

		/* Row 5 */
		va = mm256_broadcast_bf16(A[5 * lda + k]);
		c50 = _mm256_fmadd_ps(va, vb0, c50);
		c51 = _mm256_fmadd_ps(va, vb1, c51);

		k++;
	}

	/* Store C back to memory (F32) */
	_mm256_storeu_ps(&C[0 * ldc + 0], c00);
	_mm256_storeu_ps(&C[0 * ldc + 8], c01);
	_mm256_storeu_ps(&C[1 * ldc + 0], c10);
	_mm256_storeu_ps(&C[1 * ldc + 8], c11);
	_mm256_storeu_ps(&C[2 * ldc + 0], c20);
	_mm256_storeu_ps(&C[2 * ldc + 8], c21);
	_mm256_storeu_ps(&C[3 * ldc + 0], c30);
	_mm256_storeu_ps(&C[3 * ldc + 8], c31);
	_mm256_storeu_ps(&C[4 * ldc + 0], c40);
	_mm256_storeu_ps(&C[4 * ldc + 8], c41);
	_mm256_storeu_ps(&C[5 * ldc + 0], c50);
	_mm256_storeu_ps(&C[5 * ldc + 8], c51);
}

/*
** Naive scalar GEMM for comparison (BF16 -> F32)
** C[M x N] = A[M x K] * B[K x N]
*/
void	gemm_naive_bf16(
	int M, int N, int K,
	const uint16_t *A, int lda,
	const uint16_t *B, int ldb,
	float *C, int ldc)
{
	int		i;
	int		j;
	int		kk;
	float	sum;
	float	a_val;
	float	b_val;

	i = 0;
	while (i < M)
	{
		j = 0;
		while (j < N)
		{
			sum = 0.0f;
			kk = 0;
			while (kk < K)
			{
				a_val = bf16_to_f32_safe(A[i * lda + kk]);
				b_val = bf16_to_f32_safe(B[kk * ldb + j]);
				sum += a_val * b_val;
				kk++;
			}
			C[i * ldc + j] = sum;
			j++;
		}
		i++;
	}
}
