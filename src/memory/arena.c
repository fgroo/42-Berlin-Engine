/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   arena.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "arena.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*
** SIMD ALIGNMENT FIX (Issue #4)
** 
** malloc() only guarantees 8 or 16-byte alignment on most systems.
** AVX2 instructions like _mm256_load_ps REQUIRE 32-byte alignment,
** and cache-line alignment (64 bytes) is optimal for performance.
**
** We use posix_memalign() to guarantee 64-byte alignment of the arena base.
** This ensures all subsequent arena allocations (which also align to 64)
** will be properly aligned for SIMD operations.
*/

#define ARENA_ALIGNMENT 64  /* Cache line size for modern x86 CPUs */

void	arena_init(t_arena *a, size_t size)
{
	void	*ptr;
	int		ret;

	ptr = NULL;
	ret = posix_memalign(&ptr, ARENA_ALIGNMENT, size);
	if (ret != 0 || !ptr)
	{
		fprintf(stderr, "Fatal: arena posix_memalign failed (ret=%d)\n", ret);
		exit(1);
	}
	a->base = (char *)ptr;
	a->size = size;
	a->offset = 0;
}

void	*arena_try_alloc(t_arena *a, size_t size)
{
	size_t	padding;
	void	*ptr;

	padding = (64 - (a->offset % 64)) % 64;
	if (a->offset + padding + size > a->size)
		return (NULL);
	a->offset += padding;
	ptr = a->base + a->offset;
	a->offset += size;
	memset(ptr, 0, size);
	return (ptr);
}

void	*arena_alloc(t_arena *a, size_t size)
{
	void	*ptr;

	ptr = arena_try_alloc(a, size);
	if (!ptr)
	{
		fprintf(stderr, "Fatal: arena OOM\n");
		exit(1);
	}
	return (ptr);
}

void	arena_reset(t_arena *a)
{
	a->offset = 0;
}

void	arena_free(t_arena *a)
{
	free(a->base);
	a->base = NULL;
	a->size = 0;
	a->offset = 0;
}
