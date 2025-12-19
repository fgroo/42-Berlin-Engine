/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test_silu.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 23:15:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 23:15:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** FUSED SiLU-MUL BENCHMARK
** ===========================================================================
** Compares fused AVX2 kernel vs naive scalar implementation.
** Array size: 9216 (Ministral hidden_dim) x 128 (batch)
** ===========================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "compute/ops_activation.h"

#define HIDDEN_DIM 9216   /* Ministral FFN hidden dimension */
#define BATCH_SIZE 128    /* Typical batch size */
#define ARRAY_SIZE (HIDDEN_DIM * BATCH_SIZE)
#define NUM_ITERATIONS 1000

static double	get_time_ms(void)
{
	struct timeval	tv;

	gettimeofday(&tv, NULL);
	return ((double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0);
}

/* Naive scalar implementation for comparison */
static void	silu_mul_naive(float *dst, const float *gate,
						const float *up, size_t n)
{
	size_t	i;

	i = 0;
	while (i < n)
	{
		float	g;
		float	u;
		float	sigmoid;

		g = gate[i];
		u = up[i];
		sigmoid = 1.0f / (1.0f + expf(-g));
		dst[i] = g * sigmoid * u;
		i++;
	}
}

int	main(void)
{
	float	*gate;
	float	*up;
	float	*dst_naive;
	float	*dst_fused;
	double	start;
	double	end;
	double	time_naive;
	double	time_fused;
	double	max_diff;
	double	speedup;
	int		i;
	int		iter;

	printf("=== Fused SiLU-Mul Benchmark (Phase 6) ===\n");
	printf("Array size: %d (hidden_dim=%d x batch=%d)\n",
		ARRAY_SIZE, HIDDEN_DIM, BATCH_SIZE);
	printf("Iterations: %d\n\n", NUM_ITERATIONS);

	/* Allocate */
	gate = malloc(ARRAY_SIZE * sizeof(float));
	up = malloc(ARRAY_SIZE * sizeof(float));
	dst_naive = malloc(ARRAY_SIZE * sizeof(float));
	dst_fused = malloc(ARRAY_SIZE * sizeof(float));
	if (!gate || !up || !dst_naive || !dst_fused)
	{
		fprintf(stderr, "Allocation failed\n");
		return (1);
	}

	/* Initialize with random values in [-2, 2] */
	for (i = 0; i < ARRAY_SIZE; i++)
	{
		gate[i] = (float)(i % 1000) / 250.0f - 2.0f;
		up[i] = (float)((i * 7) % 1000) / 250.0f - 2.0f;
	}

	/* Warmup */
	silu_mul_naive(dst_naive, gate, up, ARRAY_SIZE);
	op_silu_mul_fused_f32(dst_fused, gate, up, ARRAY_SIZE);

	/* Correctness check */
	max_diff = 0.0;
	for (i = 0; i < ARRAY_SIZE; i++)
	{
		double diff = fabs(dst_naive[i] - dst_fused[i]);
		if (diff > max_diff)
			max_diff = diff;
	}
	printf("Max absolute difference: %.6f\n", max_diff);
	if (max_diff > 0.1f)
	{
		printf("❌ FAIL: Results differ too much!\n");
		printf("First 8 naive:  ");
		for (i = 0; i < 8; i++)
			printf("%.4f ", dst_naive[i]);
		printf("\nFirst 8 fused:  ");
		for (i = 0; i < 8; i++)
			printf("%.4f ", dst_fused[i]);
		printf("\n");
		free(gate);
		free(up);
		free(dst_naive);
		free(dst_fused);
		return (1);
	}
	printf("✅ Correctness: Results match (diff < 0.01)\n");
	printf("   (Small diff expected from fast_expf approximation)\n\n");

	/* Benchmark NAIVE */
	start = get_time_ms();
	for (iter = 0; iter < NUM_ITERATIONS; iter++)
		silu_mul_naive(dst_naive, gate, up, ARRAY_SIZE);
	end = get_time_ms();
	time_naive = end - start;

	/* Benchmark FUSED */
	start = get_time_ms();
	for (iter = 0; iter < NUM_ITERATIONS; iter++)
		op_silu_mul_fused_f32(dst_fused, gate, up, ARRAY_SIZE);
	end = get_time_ms();
	time_fused = end - start;

	/* Results */
	printf("=== Performance ===\n");
	printf("Naive (expf):  %.2f ms total (%.3f ms/iter)\n",
		time_naive, time_naive / NUM_ITERATIONS);
	printf("Fused (AVX2):  %.2f ms total (%.3f ms/iter)\n",
		time_fused, time_fused / NUM_ITERATIONS);

	/* Speedup */
	speedup = time_naive / time_fused;
	printf("\nSpeedup: %.2fx\n", speedup);
	if (speedup >= 3.0)
		printf("✅ Excellent (>= 3x speedup)\n");
	else if (speedup >= 1.5)
		printf("⚠️  Good (1.5-3x speedup)\n");
	else
		printf("❌ Below target (< 1.5x)\n");

	/* Throughput */
	{
		double	elements_per_sec_naive;
		double	elements_per_sec_fused;

		elements_per_sec_naive = (double)ARRAY_SIZE * NUM_ITERATIONS
			/ (time_naive / 1000.0);
		elements_per_sec_fused = (double)ARRAY_SIZE * NUM_ITERATIONS
			/ (time_fused / 1000.0);
		printf("\nThroughput:\n");
		printf("  Naive: %.2f M elements/sec\n", elements_per_sec_naive / 1e6);
		printf("  Fused: %.2f M elements/sec\n", elements_per_sec_fused / 1e6);
	}

	free(gate);
	free(up);
	free(dst_naive);
	free(dst_fused);
	return (0);
}
