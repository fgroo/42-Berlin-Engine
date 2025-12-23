/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   safe_alloc.h                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 15:00:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/23 14:00:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef SAFE_ALLOC_H
# define SAFE_ALLOC_H

# include <stdlib.h>
# include <stdio.h>
# include <string.h>

/*
** ============================================================================
** [FIX #4] MEMORY SAFETY - NULL CHECKS HANDLED AT ALLOCATION
** ============================================================================
** PROBLEM: Scattered malloc() calls throughout codebase without NULL checks
** leads to undefined behavior on allocation failure.
**
** SOLUTION: Use xmalloc/xcalloc/xrealloc instead of raw malloc/calloc/realloc.
** These functions:
**   1. Check for NULL return from the underlying allocator
**   2. Print a diagnostic message to stderr
**   3. exit(1) to terminate cleanly
**
** DESIGN RATIONALE: "Fail-fast" approach
**   - LLM inference requires consistent memory (no half-initialized state)
**   - Recovery from OOM mid-inference is impractical (model is corrupted)
**   - Clean termination is preferable to silent data corruption
**   - All callers can assume non-NULL return (no defensive checks needed)
**
** USAGE: Replace malloc/calloc/realloc with xmalloc/xcalloc/xrealloc everywhere.
** If you need graceful OOM handling (e.g., sparse attention fallback),
** use arena_try_alloc() from arena.c instead.
** ============================================================================
*/

static inline void	*xmalloc(size_t size)
{
	void	*ptr;

	ptr = malloc(size);
	if (!ptr)
	{
		fprintf(stderr, "FATAL: malloc(%zu) failed - out of memory\n", size);
		exit(1);
	}
	return (ptr);
}

static inline void	*xcalloc(size_t nmemb, size_t size)
{
	void	*ptr;

	ptr = calloc(nmemb, size);
	if (!ptr)
	{
		fprintf(stderr, "FATAL: calloc(%zu, %zu) failed - out of memory\n",
			nmemb, size);
		exit(1);
	}
	return (ptr);
}

static inline void	*xrealloc(void *old_ptr, size_t size)
{
	void	*ptr;

	ptr = realloc(old_ptr, size);
	if (!ptr && size > 0)
	{
		fprintf(stderr, "FATAL: realloc(%zu) failed - out of memory\n", size);
		exit(1);
	}
	return (ptr);
}

/*
** SAFE_STRCAT: Bounded string concatenation.
** Returns bytes written (excluding null), or -1 if truncated.
** Always null-terminates.
*/
static inline int	safe_strcat(char *dest, size_t dest_size,
					const char *src, size_t *current_len)
{
	size_t	src_len;
	size_t	space_left;
	size_t	copy_len;

	if (!dest || !src || dest_size == 0)
		return (-1);
	src_len = strlen(src);
	if (*current_len >= dest_size - 1)
		return (-1);
	space_left = dest_size - *current_len - 1;
	copy_len = (src_len < space_left) ? src_len : space_left;
	memcpy(dest + *current_len, src, copy_len);
	*current_len += copy_len;
	dest[*current_len] = '\0';
	if (copy_len < src_len)
		return (-1);
	return ((int)copy_len);
}

#endif
