/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_matmul.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/14 17:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include <stdio.h>
#include <string.h>
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#define USE_SIMD 1
#else
#define USE_SIMD 0
#endif

/*
** ============================================================================
** CACHE BLOCKING PARAMETERS (Operation Heavy Metal)
** ============================================================================
** These are tuned for modern x86 CPUs with:
** - L1D: 32KB per core
** - L2: 256KB-1MB per core
** - AVX2: 16 YMM registers (256-bit)
**
** For GEMV (M=1): Focus on K-blocking to keep vec[] in L1
** For GEMM (M>1): Full 3D tiling with MC/NC/KC
*/

#define MC 64   /* Block rows of A / rows of C */
#define NC 256  /* Block cols of B / cols of C (must be multiple of 8 for AVX) */
#define KC 256  /* Block inner dimension (k-loop) */

/* Prefetch distance in bytes (256 = 4 cache lines) */
#define PREFETCH_DIST 256

/*
** ============================================================================
** GEMV: Matrix-Vector Multiply (Hot Path for Generation)
** ============================================================================
** out[m] = A[m,k] @ vec[k]
** A is BF16 weights, vec is F32 activations
**
** Optimizations:
** 1. K-blocking: Keep portions of vec[] in L1 cache
** 2. Multiple accumulators: Better ILP (Instruction-Level Parallelism)
** 3. Prefetching: Prefetch next row while computing current
*/
static void	matvec_bf16_f32_blocked(float *out, const t_bf16 *a,
				const float *vec, int m, int k)
{
#if USE_SIMD
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
	{
		const t_bf16 *row = a + i * k;
		
		/* Prefetch next row to L2 cache */
		if (i + 1 < m)
			_mm_prefetch((char *)(a + (i + 1) * k), _MM_HINT_T1);
		
		/* Use 4 accumulators for better ILP (hides FMA latency) */
		__m256 sum0 = _mm256_setzero_ps();
		__m256 sum1 = _mm256_setzero_ps();
		__m256 sum2 = _mm256_setzero_ps();
		__m256 sum3 = _mm256_setzero_ps();
		
		int j = 0;
		
		/* Main loop: Process 32 elements at a time (4x8) */
		for (; j + 31 < k; j += 32)
		{
			/* Prefetch ahead */
			_mm_prefetch((char *)(row + j + PREFETCH_DIST), _MM_HINT_T0);
			_mm_prefetch((char *)(vec + j + PREFETCH_DIST), _MM_HINT_T0);
			
			/* Load and convert 4 groups of 8 BF16 weights */
			__m128i bf16_0 = _mm_loadu_si128((__m128i *)(row + j));
			__m128i bf16_1 = _mm_loadu_si128((__m128i *)(row + j + 8));
			__m128i bf16_2 = _mm_loadu_si128((__m128i *)(row + j + 16));
			__m128i bf16_3 = _mm_loadu_si128((__m128i *)(row + j + 24));
			
			__m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(bf16_0), 16));
			__m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(bf16_1), 16));
			__m256 w2 = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(bf16_2), 16));
			__m256 w3 = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(bf16_3), 16));
			
			/* Load 4 groups of 8 F32 activations */
			__m256 v0 = _mm256_loadu_ps(vec + j);
			__m256 v1 = _mm256_loadu_ps(vec + j + 8);
			__m256 v2 = _mm256_loadu_ps(vec + j + 16);
			__m256 v3 = _mm256_loadu_ps(vec + j + 24);
			
			/* FMA into separate accumulators */
			sum0 = _mm256_fmadd_ps(w0, v0, sum0);
			sum1 = _mm256_fmadd_ps(w1, v1, sum1);
			sum2 = _mm256_fmadd_ps(w2, v2, sum2);
			sum3 = _mm256_fmadd_ps(w3, v3, sum3);
		}
		
		/* Cleanup loop: 8 elements at a time */
		for (; j + 7 < k; j += 8)
		{
			__m128i bf16_vals = _mm_loadu_si128((__m128i *)(row + j));
			__m256 w = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(bf16_vals), 16));
			__m256 v = _mm256_loadu_ps(vec + j);
			sum0 = _mm256_fmadd_ps(w, v, sum0);
		}
		
		/* Combine accumulators */
		sum0 = _mm256_add_ps(sum0, sum1);
		sum2 = _mm256_add_ps(sum2, sum3);
		sum0 = _mm256_add_ps(sum0, sum2);
		
		/* Horizontal sum */
		__m128 lo = _mm256_castps256_ps128(sum0);
		__m128 hi = _mm256_extractf128_ps(sum0, 1);
		lo = _mm_add_ps(lo, hi);
		lo = _mm_hadd_ps(lo, lo);
		lo = _mm_hadd_ps(lo, lo);
		float sum = _mm_cvtss_f32(lo);
		
		/* Scalar remainder */
		for (; j < k; j++)
			sum += bf16_to_float(row[j]) * vec[j];
		
		out[i] = sum;
	}
#else
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
	{
		float sum = 0.0f;
		const t_bf16 *row = a + i * k;
		for (int j = 0; j < k; j++)
			sum += bf16_to_float(row[j]) * vec[j];
		out[i] = sum;
	}
#endif
}

/*
** F32 version of blocked GEMV
*/
static void	matvec_f32_f32_blocked(float *out, const float *a,
				const float *vec, int m, int k)
{
#if USE_SIMD
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
	{
		const float *row = a + i * k;
		
		__m256 sum0 = _mm256_setzero_ps();
		__m256 sum1 = _mm256_setzero_ps();
		__m256 sum2 = _mm256_setzero_ps();
		__m256 sum3 = _mm256_setzero_ps();
		
		int j = 0;
		for (; j + 31 < k; j += 32)
		{
			_mm_prefetch((char *)(row + j + PREFETCH_DIST), _MM_HINT_T0);
			
			sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j),
				_mm256_loadu_ps(vec + j), sum0);
			sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j + 8),
				_mm256_loadu_ps(vec + j + 8), sum1);
			sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j + 16),
				_mm256_loadu_ps(vec + j + 16), sum2);
			sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j + 24),
				_mm256_loadu_ps(vec + j + 24), sum3);
		}
		
		for (; j + 7 < k; j += 8)
			sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j),
				_mm256_loadu_ps(vec + j), sum0);
		
		sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1),
			_mm256_add_ps(sum2, sum3));
		
		__m128 lo = _mm256_castps256_ps128(sum0);
		__m128 hi = _mm256_extractf128_ps(sum0, 1);
		lo = _mm_add_ps(lo, hi);
		lo = _mm_hadd_ps(lo, lo);
		lo = _mm_hadd_ps(lo, lo);
		float sum = _mm_cvtss_f32(lo);
		
		for (; j < k; j++)
			sum += row[j] * vec[j];
		
		out[i] = sum;
	}
#else
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
	{
		float sum = 0.0f;
		const float *row = a + i * k;
		for (int j = 0; j < k; j++)
			sum += row[j] * vec[j];
		out[i] = sum;
	}
#endif
}

/*
** ============================================================================
** GEMM: Matrix-Matrix Multiply with Cache Blocking (Rare but needed)
** ============================================================================
** C[m,n] = A[m,k] @ B[k,n]
** A is BF16 weights, B is F32 activations, C is F32
**
** Uses 3D tiling: MC x NC x KC blocks
** Each tile fits in L1/L2 cache for reuse
*/

#if USE_SIMD
static void	gemm_bf16_f32_tiled(float *c, const t_bf16 *a, const float *b,
				int m, int n, int k)
{
	/* Zero output */
	memset(c, 0, m * n * sizeof(float));
	
	/* Outer loops: tile over m, n, k */
	#pragma omp parallel for collapse(2) schedule(static)
	for (int i0 = 0; i0 < m; i0 += MC)
	{
		for (int j0 = 0; j0 < n; j0 += NC)
		{
			int i_max = (i0 + MC > m) ? m : i0 + MC;
			int j_max = (j0 + NC > n) ? n : j0 + NC;
			
			/* K-loop: accumulate into C tile */
			for (int k0 = 0; k0 < k; k0 += KC)
			{
				int k_max = (k0 + KC > k) ? k : k0 + KC;
				
				/* Micro-kernel: compute one MC x NC tile */
				for (int ii = i0; ii < i_max; ii++)
				{
					const t_bf16 *a_row = a + ii * k + k0;
					float *c_row = c + ii * n + j0;
					
					for (int kk = 0; kk < k_max - k0; kk++)
					{
						/* Broadcast A value */
						float a_val = bf16_to_float(a_row[kk]);
						__m256 a_vec = _mm256_set1_ps(a_val);
						
						const float *b_row = b + (k0 + kk) * n + j0;
						
						/* Inner loop: vectorized over j */
						int jj = 0;
						for (; jj + 7 < j_max - j0; jj += 8)
						{
							__m256 c_vec = _mm256_loadu_ps(c_row + jj);
							__m256 b_vec = _mm256_loadu_ps(b_row + jj);
							c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
							_mm256_storeu_ps(c_row + jj, c_vec);
						}
						
						/* Scalar remainder */
						for (; jj < j_max - j0; jj++)
							c_row[jj] += a_val * b_row[jj];
					}
				}
			}
		}
	}
}
#endif

/*
** ============================================================================
** MAIN ENTRY POINT
** ============================================================================
*/
int	op_matmul(t_tensor *out, const t_tensor *a, const t_tensor *b)
{
	int	m, k, n;

	m = a->shape[0];
	k = a->shape[1];
	n = b->shape[1];

	if (a->shape[1] != b->shape[0])
	{
		fprintf(stderr, "MatMul dim mismatch: [%d,%d] x [%d,%d]\n",
			a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
		return (-1);
	}

	/* Case 1: A=BF16, B=F32, Out=F32 (Weights * Activations) */
	if (a->dtype == DTYPE_BF16 && b->dtype == DTYPE_F32
		&& out->dtype == DTYPE_F32)
	{
		if (n == 1)
		{
			/* GEMV: Use blocked version with prefetching */
			matvec_bf16_f32_blocked((float *)out->data,
				(t_bf16 *)a->data, (float *)b->data, m, k);
		}
		else
		{
#if USE_SIMD
			/* GEMM: Use tiled version */
			gemm_bf16_f32_tiled((float *)out->data,
				(t_bf16 *)a->data, (float *)b->data, m, n, k);
#else
			/* Fallback: naive GEMM */
			float *out_data = (float *)out->data;
			t_bf16 *a_data = (t_bf16 *)a->data;
			float *b_data = (float *)b->data;
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < n; j++)
				{
					float sum = 0.0f;
					for (int l = 0; l < k; l++)
						sum += bf16_to_float(a_data[i * k + l])
							* b_data[l * n + j];
					out_data[i * n + j] = sum;
				}
			}
#endif
		}
	}
	/* Case 2: A=BF16, B=BF16, Out=BF16 */
	else if (a->dtype == DTYPE_BF16 && b->dtype == DTYPE_BF16
		&& out->dtype == DTYPE_BF16)
	{
		t_bf16 *out_data = (t_bf16 *)out->data;
		t_bf16 *a_data = (t_bf16 *)a->data;
		t_bf16 *b_data = (t_bf16 *)b->data;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				float sum = 0.0f;
				for (int l = 0; l < k; l++)
					sum += bf16_to_float(a_data[i * k + l])
						* bf16_to_float(b_data[l * n + j]);
				out_data[i * n + j] = float_to_bf16(sum);
			}
		}
	}
	/* Case 3: A=F32, B=F32, Out=F32 */
	else if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32
		&& out->dtype == DTYPE_F32)
	{
		if (n == 1)
		{
			matvec_f32_f32_blocked((float *)out->data,
				(float *)a->data, (float *)b->data, m, k);
		}
		else
		{
			/* Naive GEMM for F32 (rare path) */
			float *out_data = (float *)out->data;
			float *a_data = (float *)a->data;
			float *b_data = (float *)b->data;
			#pragma omp parallel for collapse(2)
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < n; j++)
				{
					float sum = 0.0f;
					for (int l = 0; l < k; l++)
						sum += a_data[i * k + l] * b_data[l * n + j];
					out_data[i * n + j] = sum;
				}
			}
		}
	}
	else
	{
		fprintf(stderr, "MatMul unsupported dtypes: A=%d B=%d Out=%d\n",
			a->dtype, b->dtype, out->dtype);
		return (-1);
	}
	return (0);
}
