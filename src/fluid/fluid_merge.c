/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid_merge.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 19:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 19:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** FLUID-MERGE: The Neural Linker
** ===========================================================================
** Merges multiple .fluid capsules into one unified knowledge base.
** 
** Algorithm:
**   1. Load all input files into memory
**   2. Sort entries by context_hash (brings collisions together)
**   3. Resolve conflicts using max-weight-wins strategy
**   4. Write deduplicated output
**
** Complexity: O(N log N) where N = total entries across all files
** ===========================================================================
*/

#include "../fluid/fluid_spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
** Dynamic buffer for collecting entries from all input files
*/
typedef struct s_entry_buffer
{
	t_fluid_entry	*data;
	size_t			count;
	size_t			capacity;
}	t_entry_buffer;

/*
** Add entries to buffer, growing as needed
*/
static void	buf_add(t_entry_buffer *buf, t_fluid_entry *entries, size_t n)
{
	size_t	new_capacity;

	if (buf->count + n > buf->capacity)
	{
		new_capacity = (buf->count + n) * 2 + 1024;
		buf->data = realloc(buf->data, new_capacity * sizeof(t_fluid_entry));
		if (!buf->data)
		{
			fprintf(stderr, "[LINKER] Out of memory!\n");
			exit(1);
		}
		buf->capacity = new_capacity;
	}
	memcpy(buf->data + buf->count, entries, n * sizeof(t_fluid_entry));
	buf->count += n;
}

/*
** Qsort comparator: Sort by hash first, then by weight (descending)
** This ensures when we dedupe, the strongest weight is first
*/
static int	cmp_entries(const void *a, const void *b)
{
	const t_fluid_entry	*ea = (const t_fluid_entry *)a;
	const t_fluid_entry	*eb = (const t_fluid_entry *)b;

	if (ea->context_hash < eb->context_hash)
		return (-1);
	if (ea->context_hash > eb->context_hash)
		return (1);
	/* Same hash: higher weight wins (comes first) */
	if (ea->weight > eb->weight)
		return (-1);
	if (ea->weight < eb->weight)
		return (1);
	return (0);
}

/*
** Load a single .fluid file and add its entries to the buffer
*/
static int	load_fluid_file(const char *path, t_entry_buffer *buf,
				uint64_t *base_hash, int *first_file, char *merged_domain)
{
	FILE			*f;
	t_fluid_header	h;
	t_fluid_entry	*entries;

	f = fopen(path, "rb");
	if (!f)
	{
		fprintf(stderr, "[LINKER] Cannot open: %s\n", path);
		return (0);
	}
	if (fread(&h, sizeof(h), 1, f) != 1)
	{
		fclose(f);
		return (0);
	}
	/* Validate magic */
	if (memcmp(h.magic, FLUID_MAGIC, 4) != 0)
	{
		fprintf(stderr, "[LINKER] Skipping %s: Invalid magic\n", path);
		fclose(f);
		return (0);
	}
	/* Base model compatibility check */
	if (*first_file)
	{
		*base_hash = h.base_model_hash;
		*first_file = 0;
	}
	else if (h.base_model_hash != 0 && *base_hash != 0 
		&& h.base_model_hash != *base_hash)
	{
		fprintf(stderr, "[LINKER] FATAL: %s is for a different base model!\n", path);
		fclose(f);
		return (-1);
	}
	/* Build merged domain string */
	if (h.domain[0] && strlen(merged_domain) + strlen(h.domain) + 2 < FLUID_META_STR_LEN)
	{
		if (merged_domain[0])
			strcat(merged_domain, "+");
		strcat(merged_domain, h.domain);
	}
	/* Load entries */
	if (h.n_entries == 0)
	{
		fclose(f);
		printf("  + %s: 0 patterns (empty)\n", path);
		return (1);
	}
	entries = malloc(h.n_entries * sizeof(t_fluid_entry));
	if (!entries)
	{
		fclose(f);
		return (0);
	}
	if (fread(entries, sizeof(t_fluid_entry), h.n_entries, f) != h.n_entries)
	{
		free(entries);
		fclose(f);
		return (0);
	}
	fclose(f);
	buf_add(buf, entries, h.n_entries);
	free(entries);
	printf("  + %s: %u patterns [%s]\n", path, h.n_entries, 
		h.domain[0] ? h.domain : "unknown");
	return (1);
}

/*
** Merge entries in-place, resolving collisions
** Returns the unique count after deduplication
*/
static size_t	merge_entries(t_entry_buffer *buf, size_t *collisions)
{
	size_t			unique_idx;
	size_t			i;
	t_fluid_entry	*data;

	if (buf->count == 0)
		return (0);
	/* Sort brings matching hashes together */
	qsort(buf->data, buf->count, sizeof(t_fluid_entry), cmp_entries);
	data = buf->data;
	unique_idx = 0;
	*collisions = 0;
	for (i = 0; i < buf->count; i++)
	{
		/* First entry or new hash -> keep it */
		if (unique_idx == 0 || data[i].context_hash != data[unique_idx - 1].context_hash)
		{
			data[unique_idx++] = data[i];
		}
		else
		{
			/* COLLISION: Same hash, different entry */
			(*collisions)++;
			/* Conflict resolution strategy: Max-Weight Wins */
			/* Already sorted by weight desc, so we skip weaker ones */
			/* Optional: Average weights for reinforcement learning */
			/* data[unique_idx-1].weight = 
			     (data[unique_idx-1].weight + data[i].weight) / 2.0f; */
		}
	}
	return (unique_idx);
}

/*
** Write the merged output file
*/
static int	write_output(const char *path, t_fluid_entry *entries,
				size_t count, uint64_t base_hash, const char *domain)
{
	FILE			*f;
	t_fluid_header	h;

	f = fopen(path, "wb");
	if (!f)
	{
		fprintf(stderr, "[LINKER] Cannot create: %s\n", path);
		return (0);
	}
	/* Build header */
	memset(&h, 0, sizeof(h));
	memcpy(h.magic, FLUID_MAGIC, 4);
	h.version = FLUID_VERSION;
	h.created_at = (uint64_t)time(NULL);
	h.base_model_hash = base_hash;
	h.n_entries = count;
	h.flags = FLUID_FLAG_VERIFIED;  /* Mark as merged & validated */
	strncpy(h.domain, domain, FLUID_META_STR_LEN - 1);
	strncpy(h.author, "Neural-Linker", FLUID_META_STR_LEN - 1);
	strncpy(h.description, "Merged Knowledge Capsule - Multiple Skills Combined",
		FLUID_DESC_LEN - 1);
	/* Write */
	if (fwrite(&h, sizeof(h), 1, f) != 1)
	{
		fclose(f);
		return (0);
	}
	if (count > 0 && fwrite(entries, sizeof(t_fluid_entry), count, f) != count)
	{
		fclose(f);
		return (0);
	}
	fclose(f);
	return (1);
}

static void	print_banner(void)
{
	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║           FLUID-MERGE: The Neural Linker                     ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");
}

int	main(int argc, char **argv)
{
	t_entry_buffer	buffer;
	uint64_t		base_hash;
	int				first_file;
	char			merged_domain[FLUID_META_STR_LEN];
	int				i;
	int				loaded;
	int				ret;
	size_t			unique_count;
	size_t			collisions = 0;

	if (argc < 4)
	{
		printf("Usage: %s <output.fluid> <input1.fluid> <input2.fluid> [...]\n\n", argv[0]);
		printf("Merges multiple .fluid capsules into one unified knowledge base.\n");
		printf("Conflicts are resolved using max-weight-wins strategy.\n");
		return (1);
	}
	print_banner();
	/* Initialize */
	memset(&buffer, 0, sizeof(buffer));
	base_hash = 0;
	first_file = 1;
	merged_domain[0] = '\0';
	loaded = 0;
	/* Load all input files */
	printf("[LINKER] Loading input capsules...\n");
	for (i = 2; i < argc; i++)
	{
		ret = load_fluid_file(argv[i], &buffer, &base_hash, &first_file, merged_domain);
		if (ret < 0)
		{
			free(buffer.data);
			return (1);
		}
		if (ret > 0)
			loaded++;
	}
	if (loaded == 0)
	{
		fprintf(stderr, "[LINKER] No valid input files!\n");
		free(buffer.data);
		return (1);
	}
	printf("\n[LINKER] Loaded %d capsules, %zu total patterns.\n", loaded, buffer.count);
	/* Merge (deduplicate + conflict resolution) */
	printf("[LINKER] Resolving conflicts...\n");
	unique_count = merge_entries(&buffer, &collisions);
	printf("[LINKER] Merge complete:\n");
	printf("  - Input:     %zu patterns\n", buffer.count);
	printf("  - Output:    %zu patterns\n", unique_count);
	printf("  - Collisions: %zu (resolved via max-weight)\n", collisions);
	printf("  - Reduction: %.1f%%\n", 
		buffer.count > 0 ? (1.0 - (double)unique_count / buffer.count) * 100 : 0);
	/* Write output */
	printf("\n[LINKER] Writing %s...\n", argv[1]);
	if (!write_output(argv[1], buffer.data, unique_count, base_hash, merged_domain))
	{
		fprintf(stderr, "[LINKER] Failed to write output!\n");
		free(buffer.data);
		return (1);
	}
	printf("\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("       SUCCESS: %zu skills merged into %s\n", unique_count, argv[1]);
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("\n");
	free(buffer.data);
	return (0);
}
