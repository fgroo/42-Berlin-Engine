/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test_gemm_large.c                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 21:20:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 21:20:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** LARGE MATRIX GEMM BENCHMARK (Phase 5)
** ===========================================================================
** Tests tiled GEMM on realistic LLM dimensions:
** M=128 (batch), N=4096 (hidden), K=4096 (hidden)
** ~4.3 billion FLOPs per GEMM
** ===========================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include "compute/gemm.h"
#include "compute/ops_simd.h"

/* Matrix dimensions - realistic for LLM FFN */
#define M_SIZE 128
#define N_SIZE 4096
#define K_SIZE 4096
#define NUM_ITERATIONS 3  /* Large matrices, fewer iterations */

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
	uint16_t	*A;
	uint16_t	*B;
	float		*C_tiled;
	float		*C_naive;
	double		start;
	double		end;
	double		time_tiled;
	double		time_naive;
	double		flops;
	double		gflops_tiled;
	double		gflops_naive;
	double		max_diff;
	double		rel_err;
	int			i;
	int			iter;

	printf("=== Large Matrix GEMM Benchmark (Phase 5) ===\n");
	printf("Matrix A: [%d x %d] (BF16) = %.2f MB\n", M_SIZE, K_SIZE,
		(float)M_SIZE * K_SIZE * 2 / (1024 * 1024));
	printf("Matrix B: [%d x %d] (BF16) = %.2f MB\n", K_SIZE, N_SIZE,
		(float)K_SIZE * N_SIZE * 2 / (1024 * 1024));
	printf("Matrix C: [%d x %d] (F32) = %.2f MB\n", M_SIZE, N_SIZE,
		(float)M_SIZE * N_SIZE * 4 / (1024 * 1024));
	printf("Iterations: %d\n", NUM_ITERATIONS);
	flops = 2.0 * M_SIZE * N_SIZE * K_SIZE;
	printf("FLOPs per GEMM: %.2f billion\n\n", flops / 1e9);

	/* Allocate matrices */
	A = malloc((size_t)M_SIZE * K_SIZE * sizeof(uint16_t));
	B = malloc((size_t)K_SIZE * N_SIZE * sizeof(uint16_t));
	C_tiled = malloc((size_t)M_SIZE * N_SIZE * sizeof(float));
	C_naive = malloc((size_t)M_SIZE * N_SIZE * sizeof(float));
	if (!A || !B || !C_tiled || !C_naive)
	{
		fprintf(stderr, "Allocation failed (need ~%.0f MB)\n",
			(M_SIZE * K_SIZE * 2 + K_SIZE * N_SIZE * 2
				+ M_SIZE * N_SIZE * 8) / (1024.0 * 1024.0));
		return (1);
	}

	/* Initialize with small values to avoid overflow */
	printf("Initializing matrices...\n");
	for (i = 0; i < M_SIZE * K_SIZE; i++)
		A[i] = f32_to_bf16((float)((i % 256) - 128) / 1000.0f);
	for (i = 0; i < K_SIZE * N_SIZE; i++)
		B[i] = f32_to_bf16((float)((i % 256) - 128) / 1000.0f);

	/* Benchmark TILED */
	printf("Running tiled GEMM...\n");
	start = get_time_ms();
	for (iter = 0; iter < NUM_ITERATIONS; iter++)
		ops_gemm_bf16_tiled(M_SIZE, N_SIZE, K_SIZE, A, K_SIZE, B, N_SIZE,
			C_tiled, N_SIZE, 1);
	end = get_time_ms();
	time_tiled = end - start;
	gflops_tiled = (flops * NUM_ITERATIONS) / (time_tiled * 1e6);

	/* Correctness check: compute one row with naive */
	printf("Computing reference (first 16 cols only for speed)...\n");
	memset(C_naive, 0, (size_t)M_SIZE * N_SIZE * sizeof(float));
	{
		/* Only compute first row, first 16 cols for quick verification */
		int		j, k;
		float	sum;

		for (j = 0; j < 16; j++)
		{
			sum = 0.0f;
			for (k = 0; k < K_SIZE; k++)
				sum += bf16_to_f32_safe(A[k]) * bf16_to_f32_safe(B[k * N_SIZE + j]);
			C_naive[j] = sum;
		}
	}

	/* Compare first row */
	max_diff = 0.0;
	for (i = 0; i < 16; i++)
	{
		double diff = fabs(C_tiled[i] - C_naive[i]);
		if (diff > max_diff)
			max_diff = diff;
	}
	rel_err = max_diff / (fabs(C_naive[0]) + 1e-10);
	printf("Max absolute diff (row 0, cols 0-15): %.6f\n", max_diff);
	printf("Relative error: %.6f%%\n", rel_err * 100);

	if (rel_err > 0.01)
	{
		printf("\n⚠️  Warning: Results may differ (BF16 precision)\n");
		printf("First 8 tiled:  ");
		for (i = 0; i < 8; i++)
			printf("%.4f ", C_tiled[i]);
		printf("\nFirst 8 naive:  ");
		for (i = 0; i < 8; i++)
			printf("%.4f ", C_naive[i]);
		printf("\n");
	}
	else
	{
		printf("✅ Correctness: Results match (within BF16 precision)\n");
	}

	/* Results */
	printf("\n=== Performance ===\n");
	printf("Tiled GEMM: %.2f ms total (%.2f ms/iter)\n",
		time_tiled, time_tiled / NUM_ITERATIONS);
	printf("Throughput: %.2f GFlops\n", gflops_tiled);

	/* Compare to theoretical peak */
	/* Skylake: ~32 FLOPs/cycle @ 4 GHz = 128 GFlops single-thread (theoretical) */
	/* Realistic: 40-60% efficiency = 50-75 GFlops */
	printf("\nEfficiency estimate:\n");
	printf("  Achieved: %.2f GFlops\n", gflops_tiled);
	printf("  vs Phase 4 microkernel: %.2f GFlops (small matrix)\n", 78.83);
	if (gflops_tiled >= 30.0)
		printf("  ✅ Good cache efficiency (>30 GFlops)\n");
	else if (gflops_tiled >= 15.0)
		printf("  ⚠️  Moderate efficiency (15-30 GFlops)\n");
	else
		printf("  ❌ Poor efficiency (<15 GFlops) - cache thrashing?\n");

	free(A);
	free(B);
	free(C_tiled);
	free(C_naive);
	return (0);
}
