/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ops_matmul.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ops_internal.h"
#include <stdio.h>
#include <omp.h>  // OpenMP parallelization

// TEMPORARILY FORCE SCALAR MODE TO DEBUG AVX2 BUG
#if 1  // Re-enabled AVX2 for performance
#ifdef __AVX2__
#include <immintrin.h>
#define USE_SIMD 1
#else
#define USE_SIMD 0
#endif
#else
#define USE_SIMD 0  // Force scalar for debugging
#endif

/*
** Matrix-Vector multiply: out[m] = A[m,k] x vec[k]
** This is the hot path for inference (batch size 1)
** A is BF16 weights, vec is F32 activations
*/
static void	matvec_bf16_f32(float *out, const t_bf16 *a, const float *vec,
				int m, int k)
{
#if USE_SIMD
	// AVX2 version: process 8 floats at a time (PARALLEL)
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
	{
		const t_bf16 *row = a + i * k;
		__m256 sum_vec = _mm256_setzero_ps();
		int j;
		
		// Process 8 elements at a time
		for (j = 0; j + 7 < k; j += 8)
		{
			// Load 8 BF16 values and convert to F32
			// OPTIMIZED: cvtepu16_epi32 + slli (2 ops vs 4+ ops)
			// BF16 is upper 16 bits of F32, so shift left by 16
			__m128i bf16_vals = _mm_loadu_si128((__m128i*)(row + j));
			__m256i bf16_32 = _mm256_cvtepu16_epi32(bf16_vals);
			__m256 f32_lo = _mm256_castsi256_ps(_mm256_slli_epi32(bf16_32, 16));
			
			// Load 8 F32 from vec
			__m256 vec_vals = _mm256_loadu_ps(vec + j);
			
			// FMA: sum += a * b
			sum_vec = _mm256_fmadd_ps(f32_lo, vec_vals, sum_vec);
		}
		
		// Horizontal sum of sum_vec
		__m128 lo128 = _mm256_castps256_ps128(sum_vec);
		__m128 hi128 = _mm256_extractf128_ps(sum_vec, 1);
		lo128 = _mm_add_ps(lo128, hi128);
		lo128 = _mm_hadd_ps(lo128, lo128);
		lo128 = _mm_hadd_ps(lo128, lo128);
		float sum = _mm_cvtss_f32(lo128);
		
		// Handle remaining elements
		while (j < k)
		{
			sum += bf16_to_float(row[j]) * vec[j];
			j++;
		}
		out[i] = sum;
	}
#else
	// Fallback: scalar version (PARALLEL)
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
	{
		float sum = 0.0f;
		const t_bf16 *row = a + i * k;
		for (int j = 0; j < k; j++)
		{
			sum += bf16_to_float(row[j]) * vec[j];
		}
		out[i] = sum;
	}
#endif
}

/*
** Matrix-Vector multiply: out[m] = A[m,k] x vec[k]
** Both A and vec are F32
*/
static void	matvec_f32_f32(float *out, const float *a, const float *vec,
				int m, int k)
{
	int	i;
	int	j;

#if USE_SIMD
	for (i = 0; i < m; i++)
	{
		const float *row = a + i * k;
		__m256 sum_vec = _mm256_setzero_ps();
		
		for (j = 0; j + 7 < k; j += 8)
		{
			__m256 a_vals = _mm256_loadu_ps(row + j);
			__m256 v_vals = _mm256_loadu_ps(vec + j);
			sum_vec = _mm256_fmadd_ps(a_vals, v_vals, sum_vec);
		}
		
		// Horizontal sum
		__m128 lo = _mm256_castps256_ps128(sum_vec);
		__m128 hi = _mm256_extractf128_ps(sum_vec, 1);
		lo = _mm_add_ps(lo, hi);
		lo = _mm_hadd_ps(lo, lo);
		lo = _mm_hadd_ps(lo, lo);
		float sum = _mm_cvtss_f32(lo);
		
		while (j < k)
		{
			sum += row[j] * vec[j];
			j++;
		}
		out[i] = sum;
	}
#else
	for (i = 0; i < m; i++)
	{
		float sum = 0.0f;
		const float *row = a + i * k;
		for (j = 0; j < k; j++)
		{
			sum += row[j] * vec[j];
		}
		out[i] = sum;
	}
#endif
}

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

	// Case 1: A=BF16, B=F32, Out=F32 (Weights * Activations)
	// This is the hot path for inference
	if (a->dtype == DTYPE_BF16 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32)
	{
		// For matrix-vector (n==1), use optimized path
		if (n == 1)
		{
			matvec_bf16_f32((float *)out->data, (t_bf16 *)a->data,
				(float *)b->data, m, k);
		}
		else
		{
			// General matrix-matrix (rare in inference)
			float *out_data = (float *)out->data;
			t_bf16 *a_data = (t_bf16 *)a->data;
			float *b_data = (float *)b->data;
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < n; j++)
				{
					float sum = 0.0f;
					for (int l = 0; l < k; l++)
						sum += bf16_to_float(a_data[i * k + l]) * b_data[l * n + j];
					out_data[i * n + j] = sum;
				}
			}
		}
	}
	// Case 2: A=BF16, B=BF16, Out=BF16
	else if (a->dtype == DTYPE_BF16 && b->dtype == DTYPE_BF16 && out->dtype == DTYPE_BF16)
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
					sum += bf16_to_float(a_data[i * k + l]) * bf16_to_float(b_data[l * n + j]);
				out_data[i * n + j] = float_to_bf16(sum);
			}
		}
	}
	// Case 3: A=F32, B=F32, Out=F32
	else if (a->dtype == DTYPE_F32 && b->dtype == DTYPE_F32 && out->dtype == DTYPE_F32)
	{
		if (n == 1)
		{
			matvec_f32_f32((float *)out->data, (float *)a->data,
				(float *)b->data, m, k);
		}
		else
		{
			float *out_data = (float *)out->data;
			float *a_data = (float *)a->data;
			float *b_data = (float *)b->data;
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
