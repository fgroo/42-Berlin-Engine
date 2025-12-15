/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   gemm.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 21:20:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 21:20:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** TILED GEMM DRIVER (Phase 5: L2 Cache Blocking)
** ===========================================================================
** C[M x N] = A[M x K] * B[K x N]   (BF16 inputs, F32 output)
**
** Tiling strategy:
** - MC = 6  (fixed by microkernel)
** - NC = 16 (fixed by microkernel)
** - KC = 256 (L1 cache friendly for K-dimension)
**
** Loop order: J-I-K (column-major access pattern for cache efficiency)
** Edge cases: scalar fallback for M % 6 != 0 or N % 16 != 0
** ===========================================================================
*/

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>
#include "ops_simd.h"
#include "gemm_kernel.h"

/* Block sizes */
#define MC 6
#define NC 16
#define KC 256

/*
** Scalar fallback for edge blocks (M < 6 or N < 16)
** Accumulates into C (assumes C is pre-zeroed or contains partial sums)
*/
static void	gemm_edge_scalar(
	int m_size, int n_size, int k_size,
	const uint16_t *A, int lda,
	const uint16_t *B, int ldb,
	float *C, int ldc)
{
	int		i;
	int		j;
	int		k;
	float	sum;
	float	a_val;
	float	b_val;

	i = 0;
	while (i < m_size)
	{
		j = 0;
		while (j < n_size)
		{
			sum = C[i * ldc + j];  /* Load existing value for accumulation */
			k = 0;
			while (k < k_size)
			{
				a_val = bf16_to_f32_safe(A[i * lda + k]);
				b_val = bf16_to_f32_safe(B[k * ldb + j]);
				sum += a_val * b_val;
				k++;
			}
			C[i * ldc + j] = sum;
			j++;
		}
		i++;
	}
}

/*
** Tiled GEMM: C = A * B with L2 cache blocking + OpenMP parallelization
**
** @param M, N, K: Matrix dimensions
** @param A: [M x K] BF16 row-major
** @param lda: Leading dimension of A (usually K)
** @param B: [K x N] BF16 row-major
** @param ldb: Leading dimension of B (usually N)
** @param C: [M x N] F32 row-major (pre-zeroed by caller or this function)
** @param ldc: Leading dimension of C (usually N)
** @param zero_c: If non-zero, zero C before computation
*/
void	ops_gemm_bf16_tiled(
	int M, int N, int K,
	const uint16_t *A, int lda,
	const uint16_t *B, int ldb,
	float *C, int ldc,
	int zero_c)
{
	int		i;

	/* Zero C if requested (required for accumulation) */
	if (zero_c)
	{
		#pragma omp parallel for schedule(static)
		for (i = 0; i < M; i++)
			memset(&C[i * ldc], 0, (size_t)N * sizeof(float));
	}

	/* Parallelize over N-blocks (columns) - each column block is independent */
	#pragma omp parallel for schedule(dynamic, 1)
	for (int jj = 0; jj < N; jj += NC)
	{
		int		ii;
		int		kk;
		int		j_size;
		int		i_size;
		int		k_size;

		j_size = (jj + NC <= N) ? NC : (N - jj);

		/* Loop over M in MC-sized blocks (rows) */
		ii = 0;
		while (ii < M)
		{
			i_size = (ii + MC <= M) ? MC : (M - ii);

			/* Loop over K in KC-sized blocks (depth) */
			kk = 0;
			while (kk < K)
			{
				k_size = (kk + KC <= K) ? KC : (K - kk);

				/* Check if we can use the fast 6x16 microkernel */
				if (i_size == MC && j_size == NC)
				{
					/* Full block: use SIMD microkernel */
					gemm_microkernel_6x16_bf16(
						k_size,
						&A[ii * lda + kk], lda,
						&B[kk * ldb + jj], ldb,
						&C[ii * ldc + jj], ldc);
				}
				else
				{
					/* Edge block: use scalar fallback */
					gemm_edge_scalar(
						i_size, j_size, k_size,
						&A[ii * lda + kk], lda,
						&B[kk * ldb + jj], ldb,
						&C[ii * ldc + jj], ldc);
				}
				kk += KC;
			}
			ii += MC;
		}
	}
}

/*
** Naive GEMM for comparison (full matrix, no tiling)
*/
void	ops_gemm_bf16_naive(
	int M, int N, int K,
	const uint16_t *A, int lda,
	const uint16_t *B, int ldb,
	float *C, int ldc)
{
	int		i;
	int		j;
	int		k;
	float	sum;

	i = 0;
	while (i < M)
	{
		j = 0;
		while (j < N)
		{
			sum = 0.0f;
			k = 0;
			while (k < K)
			{
				sum += bf16_to_f32_safe(A[i * lda + k])
					* bf16_to_f32_safe(B[k * ldb + j]);
				k++;
			}
			C[i * ldc + j] = sum;
			j++;
		}
		i++;
	}
}
