/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   arena.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef ARENA_H
# define ARENA_H

# include <stddef.h>

typedef struct s_arena
{
	char	*base;
	size_t	size;
	size_t	offset;
}	t_arena;

void	arena_init(t_arena *a, size_t size);
void	*arena_alloc(t_arena *a, size_t size);
void	*arena_try_alloc(t_arena *a, size_t size);
void	arena_reset(t_arena *a);
void	arena_free(t_arena *a);

#endif
