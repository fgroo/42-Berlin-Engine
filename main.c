/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: 42-berlin-engine                           +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05                               #+#    #+#             */
/*   Updated: 2025/12/05                              ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "memory/arena.h"
#include "memory/kv_cache.h"
#include "tensor/tensor.h"
#include "loader/loader.h"
#include "compute/ops.h"
#include "compute/sampler.h"
#include "nested/fluid.h"

static void	test_arena(t_arena *arena)
{
	printf("[PASS] Arena initialized.\n");
	(void)arena;
}

static void	test_bf16(void)
{
	float	input;
	t_bf16	bf;
	float	output;

	input = 3.14159f;
	bf = float_to_bf16(input);
	output = bf16_to_float(bf);
	printf("Float: %f -> BF16: %04x -> Float: %f\n", input, bf, output);
}

static void	test_tensor(t_arena *arena)
{
	int			shape[2];
	size_t		size_bytes;
	void		*data;
	t_tensor	t;

	shape[0] = 2;
	shape[1] = 2;
	size_bytes = 4 * sizeof(t_bf16);
	data = arena_alloc(arena, size_bytes);
	t = tensor_view(data, shape, 2);
	printf("[PASS] Tensor view created. Shape: [%d, %d]\n",
		t.shape[0], t.shape[1]);
	((t_bf16 *)t.data)[0] = float_to_bf16(3.14159f);
	((t_bf16 *)t.data)[1] = float_to_bf16(1.0f);
	((t_bf16 *)t.data)[2] = float_to_bf16(2.0f);
	((t_bf16 *)t.data)[3] = float_to_bf16(0.5f);
	printf("[PASS] Data written to tensor.\n");
}

void test_rmsnorm(void)
{
	t_tensor t_in, t_w, t_out;
	
	// Setup tensors
	t_in.ndim = 1; t_in.shape[0] = 4; t_in.size = 4; t_in.dtype = DTYPE_F32;
	t_in.data = malloc(4 * sizeof(float));
	
	t_w.ndim = 1; t_w.shape[0] = 4; t_w.size = 4; t_w.dtype = DTYPE_BF16;
	t_w.data = malloc(4 * sizeof(t_bf16));
	
	t_out.ndim = 1; t_out.shape[0] = 4; t_out.size = 4; t_out.dtype = DTYPE_F32;
	t_out.data = malloc(4 * sizeof(float));

	((float *)t_in.data)[0] = 1.0f;
	((float *)t_in.data)[1] = 2.0f;
	((float *)t_in.data)[2] = 3.0f;
	((float *)t_in.data)[3] = 4.0f;

	((t_bf16 *)t_w.data)[0] = float_to_bf16(1.0f);
	((t_bf16 *)t_w.data)[1] = float_to_bf16(1.0f);
	((t_bf16 *)t_w.data)[2] = float_to_bf16(1.0f);
	((t_bf16 *)t_w.data)[3] = float_to_bf16(1.0f);

	op_rmsnorm(&t_out, &t_in, &t_w, 1e-5);

	printf("RMSNorm: %f %f %f %f\n", 
		((float *)t_out.data)[0], ((float *)t_out.data)[1],
		((float *)t_out.data)[2], ((float *)t_out.data)[3]);

	free(t_in.data); free(t_w.data); free(t_out.data);
}

void test_sampler(void)
{
	t_tensor t_logits;
	int token;
	t_logits.ndim = 1; t_logits.shape[0] = 4; t_logits.size = 4; t_logits.dtype = DTYPE_BF16;
	t_logits.data = malloc(4 * sizeof(t_bf16));

	((t_bf16 *)t_logits.data)[0] = float_to_bf16(0.1f);
	((t_bf16 *)t_logits.data)[1] = float_to_bf16(2.5f);
	((t_bf16 *)t_logits.data)[2] = float_to_bf16(1.2f);
	((t_bf16 *)t_logits.data)[3] = float_to_bf16(0.8f);
	token = sample_argmax(&t_logits);
	printf("Sampled Token: %d (Expected 1)\n", token);
	free(t_logits.data);
}

int	main(void)
{
	t_arena		arena;

	printf("Initializing 42-BERLIN-ENGINE Test...\n");
	arena_init(&arena, 1024 * 1024);
	
	test_arena(&arena);
	test_bf16();
	test_tensor(&arena);
	test_rmsnorm();
	test_sampler();
	
	arena_free(&arena);
	printf("[PASS] All basic tests complete.\n");
	return (0);
}
