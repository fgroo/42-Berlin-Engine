/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test_simd.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 14:50:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 14:50:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** SIMD BENCHMARK: Sum-of-Squares Kernel
** ===========================================================================
** Compares SIMD (AVX2+FMA) vs scalar implementation on 1M floats.
** Expected speedup: 4-8x on modern Intel/AMD CPUs.
** ===========================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "compute/ops_simd.h"

#define ARRAY_SIZE (1000000)  /* 1 Million floats = 4MB */
#define NUM_ITERATIONS 100    /* Repeat for stable timing */

static double	get_time_ms(void)
{
	struct timeval	tv;

	gettimeofday(&tv, NULL);
	return ((double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0);
}

int	main(void)
{
	float	*x;
	float	result_scalar;
	float	result_simd;
	double	start;
	double	end;
	double	time_scalar;
	double	time_simd;
	double	speedup;
	int		i;
	int		iter;

	printf("=== SIMD Sum-of-Squares Benchmark ===\n");
	printf("Array size: %d floats (%.2f MB)\n", ARRAY_SIZE,
		(float)ARRAY_SIZE * sizeof(float) / (1024 * 1024));
	printf("Iterations: %d\n\n", NUM_ITERATIONS);

	/* Allocate and initialize array with random values */
	x = malloc(ARRAY_SIZE * sizeof(float));
	if (!x)
	{
		fprintf(stderr, "Failed to allocate array\n");
		return (1);
	}
	/* Initialize with deterministic "random" values */
	for (i = 0; i < ARRAY_SIZE; i++)
		x[i] = (float)(i % 1000) / 100.0f - 5.0f;  /* Values in [-5, 5) */

	/* Warmup */
	result_scalar = ops_scalar_sum_sq(x, ARRAY_SIZE);
	result_simd = ops_simd_sum_sq(x, ARRAY_SIZE);

	/* Benchmark SCALAR */
	start = get_time_ms();
	for (iter = 0; iter < NUM_ITERATIONS; iter++)
		result_scalar = ops_scalar_sum_sq(x, ARRAY_SIZE);
	end = get_time_ms();
	time_scalar = end - start;

	/* Benchmark SIMD */
	start = get_time_ms();
	for (iter = 0; iter < NUM_ITERATIONS; iter++)
		result_simd = ops_simd_sum_sq(x, ARRAY_SIZE);
	end = get_time_ms();
	time_simd = end - start;

	/* Results */
	printf("=== Results ===\n");
	printf("Scalar sum_sq: %.6f (%.2f ms total, %.3f ms/iter)\n",
		result_scalar, time_scalar, time_scalar / NUM_ITERATIONS);
	printf("SIMD sum_sq:   %.6f (%.2f ms total, %.3f ms/iter)\n",
		result_simd, time_simd, time_simd / NUM_ITERATIONS);

	/* Correctness check - 0.1% tolerance for FP accumulation order diffs */
	if (fabsf(result_scalar - result_simd) / fabsf(result_scalar) > 1e-3f)
	{
		printf("\n❌ FAIL: Results differ significantly!\n");
		printf("   Relative error: %.6f%%\n",
			fabsf(result_scalar - result_simd) / fabsf(result_scalar) * 100);
		free(x);
		return (1);
	}
	printf("\n✅ Correctness: Results match (relative error < 0.1%%)\n");
	printf("   (Small diff expected from parallel FP accumulation order)\n");

	/* Speedup */
	speedup = time_scalar / time_simd;
	printf("\n=== Performance ===\n");
	printf("Speedup: %.2fx\n", speedup);
	if (speedup >= 4.0)
		printf("✅ Target met (>= 4x speedup)\n");
	else
		printf("⚠️  Below target (expected >= 4x)\n");

	/* Throughput */
	printf("\nThroughput:\n");
	printf("  Scalar: %.2f GFlops\n",
		(double)ARRAY_SIZE * NUM_ITERATIONS * 2 / (time_scalar * 1e6));
	printf("  SIMD:   %.2f GFlops\n",
		(double)ARRAY_SIZE * NUM_ITERATIONS * 2 / (time_simd * 1e6));

	free(x);
	return (0);
}
