/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   safe_alloc.h                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 15:00:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 15:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef SAFE_ALLOC_H
# define SAFE_ALLOC_H

# include <stdlib.h>
# include <stdio.h>
# include <string.h>

/*
** SAFE_ALLOC: Memory allocation with mandatory NULL check.
** Prints error and exits cleanly on failure.
** Usage: ptr = xmalloc(size);
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
