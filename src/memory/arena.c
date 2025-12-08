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

void	arena_init(t_arena *a, size_t size)
{
	a->base = malloc(size);
	if (!a->base)
	{
		fprintf(stderr, "Fatal: arena alloc failed\n");
		exit(1);
	}
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
