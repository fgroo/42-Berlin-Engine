/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid_info.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 19:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 19:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../fluid/fluid_spec.h"
#include "../fluid/fluid_io.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

/*
** ===========================================================================
** FLUID-INFO: The Knowledge Capsule Inspector
** ===========================================================================
** Usage: ./fluid-info <file.fluid>
**
** Displays metadata and statistics about a .fluid file without loading
** the full entry data. Like `ls -l` for knowledge.
** ===========================================================================
*/

static void	print_flags(uint32_t flags)
{
	printf("Flags:       ");
	if (flags == 0)
	{
		printf("(none)\n");
		return ;
	}
	if (flags & FLUID_FLAG_COMPRESSED)
		printf("[COMPRESSED] ");
	if (flags & FLUID_FLAG_ENCRYPTED)
		printf("[ENCRYPTED] ");
	if (flags & FLUID_FLAG_DELTA)
		printf("[DELTA] ");
	if (flags & FLUID_FLAG_VERIFIED)
		printf("[VERIFIED] ");
	printf("\n");
}

static void	print_size_human(size_t bytes)
{
	if (bytes < 1024)
		printf("%zu B", bytes);
	else if (bytes < 1024 * 1024)
		printf("%.1f KB", bytes / 1024.0);
	else
		printf("%.1f MB", bytes / (1024.0 * 1024.0));
}

int	main(int argc, char **argv)
{
	t_fluid_header	h;
	int				ret;
	size_t			data_size;
	time_t			created;

	if (argc < 2)
	{
		printf("Usage: %s <file.fluid>\n", argv[0]);
		printf("\nInspect a Fluid Knowledge Capsule without loading entries.\n");
		return (1);
	}
	ret = fluid_read_header(argv[1], &h);
	if (ret != FLUID_OK)
	{
		printf("Error: %s\n", fluid_strerror(ret));
		return (1);
	}
	/* Calculate data size */
	data_size = sizeof(t_fluid_header) + (h.n_entries * sizeof(t_fluid_entry));
	created = (time_t)h.created_at;
	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║           FLUID KNOWLEDGE CAPSULE (v%u)                       ║\n", h.version);
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");
	printf("File:        %s\n", argv[1]);
	printf("Domain:      %s\n", h.domain[0] ? h.domain : "(unspecified)");
	printf("Author:      %s\n", h.author[0] ? h.author : "(unknown)");
	printf("Created:     %s", ctime(&created));
	printf("Base Hash:   0x%016lX\n", (unsigned long)h.base_model_hash);
	printf("Entries:     %u patterns\n", h.n_entries);
	printf("Size:        ");
	print_size_human(data_size);
	printf("\n");
	print_flags(h.flags);
	printf("\n");
	if (h.description[0])
	{
		printf("Description:\n  %s\n", h.description);
		printf("\n");
	}
	printf("═══════════════════════════════════════════════════════════════\n");
	return (0);
}
