/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test_gemm.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 16:00:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 16:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** GEMM MICROKERNEL BENCHMARK
** ===========================================================================
** Tests 6x16 register-blocked kernel vs naive scalar implementation.
** Matrix sizes: A[6x256], B[256x16] (fits in L1 cache)
** Expected speedup: 4-8x (limited by BF16 conversion overhead)
** ===========================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include "compute/gemm_kernel.h"
#include "compute/ops_simd.h"

#define M_SIZE 6
#define K_SIZE 256
#define N_SIZE 16
#define NUM_ITERATIONS 10000

static double	get_time_ms(void)
{
	struct timeval	tv;

	gettimeofday(&tv, NULL);
	return ((double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0);
}

/* Convert F32 to BF16 (truncation) */
static inline uint16_t	f32_to_bf16(float f)
{
	union { uint32_t u; float f; }	conv;

	conv.f = f;
	return ((uint16_t)(conv.u >> 16));
}

int	main(void)
{
	uint16_t	*A;        /* [6 x 256] BF16 */
	uint16_t	*B;        /* [256 x 16] BF16 */
	float		*C_naive;  /* [6 x 16] F32 */
	float		*C_simd;   /* [6 x 16] F32 */
	double		start;
	double		end;
	double		time_naive;
	double		time_simd;
	double		speedup;
	double		max_diff;
	double		diff;
	int			i;
	int			j;
	int			iter;

	printf("=== GEMM 6x16 Microkernel Benchmark ===\n");
	printf("Matrix A: [%d x %d] (BF16)\n", M_SIZE, K_SIZE);
	printf("Matrix B: [%d x %d] (BF16)\n", K_SIZE, N_SIZE);
	printf("Matrix C: [%d x %d] (F32)\n", M_SIZE, N_SIZE);
	printf("Iterations: %d\n\n", NUM_ITERATIONS);

	/* Allocate matrices */
	A = malloc(M_SIZE * K_SIZE * sizeof(uint16_t));
	B = malloc(K_SIZE * N_SIZE * sizeof(uint16_t));
	C_naive = malloc(M_SIZE * N_SIZE * sizeof(float));
	C_simd = malloc(M_SIZE * N_SIZE * sizeof(float));
	if (!A || !B || !C_naive || !C_simd)
	{
		fprintf(stderr, "Allocation failed\n");
		return (1);
	}

	/* Initialize with deterministic values */
	for (i = 0; i < M_SIZE * K_SIZE; i++)
		A[i] = f32_to_bf16((float)(i % 100) / 50.0f - 1.0f);  /* [-1, 1) */
	for (i = 0; i < K_SIZE * N_SIZE; i++)
		B[i] = f32_to_bf16((float)(i % 100) / 50.0f - 1.0f);

	/* Warmup and correctness check */
	memset(C_naive, 0, M_SIZE * N_SIZE * sizeof(float));
	memset(C_simd, 0, M_SIZE * N_SIZE * sizeof(float));
	gemm_naive_bf16(M_SIZE, N_SIZE, K_SIZE, A, K_SIZE, B, N_SIZE, C_naive, N_SIZE);
	gemm_microkernel_6x16_bf16(K_SIZE, A, K_SIZE, B, N_SIZE, C_simd, N_SIZE);

	/* Check correctness */
	max_diff = 0.0;
	for (i = 0; i < M_SIZE; i++)
	{
		for (j = 0; j < N_SIZE; j++)
		{
			diff = fabs(C_naive[i * N_SIZE + j] - C_simd[i * N_SIZE + j]);
			if (diff > max_diff)
				max_diff = diff;
		}
	}
	printf("Max absolute difference: %.6f\n", max_diff);
	if (max_diff > 0.1f)
	{
		printf("❌ FAIL: Results differ too much!\n");
		printf("\nNaive C:\n");
		for (i = 0; i < M_SIZE; i++)
		{
			for (j = 0; j < N_SIZE; j++)
				printf("%.2f ", C_naive[i * N_SIZE + j]);
			printf("\n");
		}
		printf("\nSIMD C:\n");
		for (i = 0; i < M_SIZE; i++)
		{
			for (j = 0; j < N_SIZE; j++)
				printf("%.2f ", C_simd[i * N_SIZE + j]);
			printf("\n");
		}
		free(A);
		free(B);
		free(C_naive);
		free(C_simd);
		return (1);
	}
	printf("✅ Correctness: Results match\n\n");

	/* Benchmark NAIVE */
	start = get_time_ms();
	for (iter = 0; iter < NUM_ITERATIONS; iter++)
	{
		memset(C_naive, 0, M_SIZE * N_SIZE * sizeof(float));
		gemm_naive_bf16(M_SIZE, N_SIZE, K_SIZE, A, K_SIZE, B, N_SIZE,
			C_naive, N_SIZE);
	}
	end = get_time_ms();
	time_naive = end - start;

	/* Benchmark SIMD */
	start = get_time_ms();
	for (iter = 0; iter < NUM_ITERATIONS; iter++)
	{
		memset(C_simd, 0, M_SIZE * N_SIZE * sizeof(float));
		gemm_microkernel_6x16_bf16(K_SIZE, A, K_SIZE, B, N_SIZE, C_simd, N_SIZE);
	}
	end = get_time_ms();
	time_simd = end - start;

	/* Results */
	printf("=== Performance ===\n");
	printf("Naive:  %.2f ms total (%.4f us/iter)\n",
		time_naive, time_naive * 1000.0 / NUM_ITERATIONS);
	printf("SIMD:   %.2f ms total (%.4f us/iter)\n",
		time_simd, time_simd * 1000.0 / NUM_ITERATIONS);

	/* Speedup */
	speedup = time_naive / time_simd;
	printf("\nSpeedup: %.2fx\n", speedup);
	if (speedup >= 4.0)
		printf("✅ Target met (>= 4x speedup)\n");
	else if (speedup >= 2.0)
		printf("⚠️  Partial success (2-4x speedup)\n");
	else
		printf("❌ Below target\n");

	/* FLOPS calculation: 2 * M * N * K ops per GEMM */
	{
		double	flops_per_gemm;
		double	gflops_naive;
		double	gflops_simd;

		flops_per_gemm = 2.0 * M_SIZE * N_SIZE * K_SIZE;
		gflops_naive = (flops_per_gemm * NUM_ITERATIONS) / (time_naive * 1e6);
		gflops_simd = (flops_per_gemm * NUM_ITERATIONS) / (time_simd * 1e6);
		printf("\nThroughput:\n");
		printf("  Naive: %.2f GFlops\n", gflops_naive);
		printf("  SIMD:  %.2f GFlops\n", gflops_simd);
	}

	free(A);
	free(B);
	free(C_naive);
	free(C_simd);
	return (0);
}
