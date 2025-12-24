/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bench_gemm.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity                                +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/23 16:30:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/23 16:30:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** SILICON BURN TEST: FFN-Sized GEMM Benchmark
** ===========================================================================
** Simulates Feed-Forward Network MatMul: [Batch x 4096] * [4096 x 4096]
** This is the critical path for Ministral-3B and similar models.
**
** Target: >50% of theoretical CPU peak (>100 GFLOPS on modern Intel/AMD)
** ===========================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "../src/memory/arena.h"
#include "../src/compute/gemm_kernel.h"

/* FFN Workload Dimensions */
#define M_FFN 128    /* Batch size (prefill tokens) */
#define N_FFN 4096   /* Output dimension */
#define K_FFN 4096   /* Input dimension (hidden size) */
#define ITERATIONS 10

/* Microkernel tile sizes (must match gemm_kernel.c) */
#define MR 6
#define NR 16

/* Convert F32 to BF16 (truncation) */
static inline uint16_t f32_to_bf16(float f)
{
	union { uint32_t u; float f; } conv;
	conv.f = f;
	return (uint16_t)(conv.u >> 16);
}

/* Tiled GEMM calling the 6x16 microkernel - OpenMP parallelized */
static void gemm_tiled_bf16(
	int M, int N, int K,
	const uint16_t *A, int lda,
	const uint16_t *B, int ldb,
	float *C, int ldc)
{
	/* Zero C before accumulation */
	memset(C, 0, M * N * sizeof(float));
	
	/* Tile over MxN, calling 6x16 microkernel */
	/* Parallelize over M-dimension (row tiles) */
	#pragma omp parallel for schedule(static)
	for (int i = 0; i <= M - MR; i += MR) {
		for (int j = 0; j <= N - NR; j += NR) {
			gemm_microkernel_6x16_bf16(
				K,
				&A[i * lda], lda,
				&B[j], ldb,  /* B is K x N row-major */
				&C[i * ldc + j], ldc
			);
		}
	}
	
	/* Handle edge cases with scalar (if M/N not divisible by MR/NR) */
	/* Skipped for benchmark purity - assumes clean dimensions */
}

int main(void)
{
	t_arena bench_arena;
	uint16_t *A, *B;
	float *C;
	struct timespec start, end;
	double time_sec, gflops, peak_utilization;
	int iter, i;
	
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║        42-BERLIN-ENGINE: SILICON BURN TEST                   ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");
	
	printf("[CONFIG] FFN Workload: A[%d x %d] * B[%d x %d] = C[%d x %d]\n",
		M_FFN, K_FFN, K_FFN, N_FFN, M_FFN, N_FFN);
	printf("[CONFIG] Iterations: %d\n", ITERATIONS);
	printf("[CONFIG] Microkernel: 6x16 (AVX2 FMA)\n\n");

	/* 1. Arena Setup (need ~130MB for matrices) */
	printf("[INIT] Allocating arena (256 MB)...\n");
	arena_init(&bench_arena, 256 * 1024 * 1024);
	if (!bench_arena.base) {
		fprintf(stderr, "FATAL: Arena init failed\n");
		return 1;
	}

	/* 2. Allocate Matrices */
	A = arena_alloc_or_die(&bench_arena, M_FFN * K_FFN * sizeof(uint16_t));
	B = arena_alloc_or_die(&bench_arena, K_FFN * N_FFN * sizeof(uint16_t));
	C = arena_alloc_or_die(&bench_arena, M_FFN * N_FFN * sizeof(float));

	/* 3. Initialize with pseudo-random values */
	printf("[INIT] Filling matrices with noise...\n");
	for (i = 0; i < M_FFN * K_FFN; i++)
		A[i] = f32_to_bf16((float)(i % 1000) / 500.0f - 1.0f);
	for (i = 0; i < K_FFN * N_FFN; i++)
		B[i] = f32_to_bf16((float)(i % 1000) / 500.0f - 1.0f);

	/* 4. Warmup */
	printf("[RUN] Warming up...\n");
	gemm_tiled_bf16(M_FFN, N_FFN, K_FFN, A, K_FFN, B, N_FFN, C, N_FFN);

	/* 5. Timed Run */
	printf("[RUN] Benchmarking %d iterations...\n\n", ITERATIONS);
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	for (iter = 0; iter < ITERATIONS; iter++) {
		gemm_tiled_bf16(M_FFN, N_FFN, K_FFN, A, K_FFN, B, N_FFN, C, N_FFN);
	}
	
	clock_gettime(CLOCK_MONOTONIC, &end);

	/* 6. Calculate Results */
	time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
	
	/* FLOPs: 2 ops per MAC (multiply + add), M*N*K MACs per GEMM */
	double total_flops = 2.0 * M_FFN * N_FFN * K_FFN * ITERATIONS;
	gflops = (total_flops / time_sec) / 1e9;
	
	/* Assume ~200 GFLOPS theoretical peak for single core AVX2 FMA @ 3GHz */
	/* (8 floats * 2 FMA units * 3GHz / 2 (FMA = 2 ops) = ~48 GFLOPS single core theoretical) */
	/* With 8 threads: ~384 GFLOPS theoretical */
	peak_utilization = gflops / 48.0 * 100.0;  /* Single-core percentage */

	printf("═══════════════════════════════════════════════════════════════\n");
	printf("                       RESULTS\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("  Total Time:       %.4f seconds\n", time_sec);
	printf("  Throughput:       %.2f GFLOPS\n", gflops);
	printf("  Peak Utilization: %.1f%% (single-core estimate)\n", peak_utilization);
	printf("───────────────────────────────────────────────────────────────\n");
	
	if (gflops >= 100.0) {
		printf("  🚀 EXCELLENT: Approaching peak compute density!\n");
	} else if (gflops >= 50.0) {
		printf("  ✅ GOOD: >50 GFLOPS achieved.\n");
	} else if (gflops >= 20.0) {
		printf("  ⚠️  FAIR: Check vectorization (-O3 -mavx2 -mfma).\n");
	} else {
		printf("  ❌ POOR: Kernel may not be vectorizing properly.\n");
	}
	printf("═══════════════════════════════════════════════════════════════\n");

	/* Sanity check: print first output value */
	printf("\n[CHECK] C[0,0] = %.4f (should be non-zero)\n", C[0]);

	arena_free(&bench_arena);
	return 0;
}
