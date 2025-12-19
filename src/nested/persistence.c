/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   persistence.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 10:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 10:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "persistence.h"
#include "inference/inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
** ============================================================================
** FLUID STATE PERSISTENCE
** ============================================================================
** "Project Black Box" - Give the engine long-term memory.
**
** We only save the DIFFERENCE between the base model and learned state:
** - Context biases (sparse - only non-zero entries)
** - Final adapter weights (if used)
**
** This keeps files small (KB to MB) instead of duplicating the 3GB model.
** ============================================================================
*/

/*
** Count non-zero entries in context bias hashmap
*/
static int	count_active_biases(t_transformer *t)
{
	int	count;
	int	i;

	count = 0;
	if (!t->context_bias.keys)
		return (0);
	i = 0;
	while (i < t->context_bias.size)
	{
		if (t->context_bias.keys[i] != 0)
			count++;
		i++;
	}
	return (count);
}

/*
** Check if final adapter has any non-zero weights
*/
static int	has_nonzero_adapter(t_transformer *t)
{
	t_bf16	*data;
	size_t	size;
	size_t	i;

	if (!t->final_adapter || !t->final_adapter->data)
		return (0);
	data = (t_bf16 *)t->final_adapter->data;
	size = (size_t)t->config.dim * t->config.dim;
	i = 0;
	while (i < size)
	{
		if (data[i] != 0)
			return (1);
		i++;
	}
	return (0);
}

int	fluid_save(t_transformer *t, const char *path)
{
	FILE			*fp;
	t_fluid_header	header;
	t_fluid_bias_entry	entry;
	int				i;
	int				written;

	fp = fopen(path, "wb");
	if (!fp)
	{
		fprintf(stderr, "[FLUID] Error: Cannot open '%s' for writing\n", path);
		return (-1);
	}

	/* Prepare header */
	memset(&header, 0, sizeof(header));
	memcpy(header.magic, FLUID_MAGIC, 4);
	header.version = FLUID_VERSION;
	header.model_dim = t->config.dim;
	header.timestamp = (uint64_t)time(NULL);
	header.n_bias_entries = count_active_biases(t);
	header.has_adapter = has_nonzero_adapter(t);

	/* Write header */
	if (fwrite(&header, sizeof(header), 1, fp) != 1)
	{
		fprintf(stderr, "[FLUID] Error: Failed to write header\n");
		fclose(fp);
		return (-1);
	}

	/* Write context bias entries (sparse - only non-zero) */
	written = 0;
	if (t->context_bias.keys)
	{
		i = 0;
		while (i < t->context_bias.size)
		{
			if (t->context_bias.keys[i] != 0)
			{
				entry.key = t->context_bias.keys[i];
				entry.target_token = t->context_bias.tokens[i];
				entry.bias = t->context_bias.biases[i];
				if (fwrite(&entry, sizeof(entry), 1, fp) != 1)
				{
					fprintf(stderr, "[FLUID] Error: Failed to write bias entry\n");
					fclose(fp);
					return (-1);
				}
				written++;
			}
			i++;
		}
	}

	/* Write final adapter if present */
	if (header.has_adapter)
	{
		size_t adapter_size = (size_t)t->config.dim * t->config.dim * sizeof(t_bf16);
		if (fwrite(t->final_adapter->data, adapter_size, 1, fp) != 1)
		{
			fprintf(stderr, "[FLUID] Error: Failed to write adapter\n");
			fclose(fp);
			return (-1);
		}
	}

	fclose(fp);
	printf("[FLUID] Saved: %d bias entries, adapter=%s, size=%ldB\n",
		written, header.has_adapter ? "yes" : "no",
		(long)(sizeof(header) + written * sizeof(entry) + 
			(header.has_adapter ? (size_t)t->config.dim * t->config.dim * 2 : 0)));
	return (0);
}

int	fluid_load(t_transformer *t, const char *path)
{
	FILE			*fp;
	t_fluid_header	header;
	t_fluid_bias_entry	entry;
	uint32_t		i;
	int				loaded;
	uint64_t		h;
	uint32_t		idx;
	int				j;

	fp = fopen(path, "rb");
	if (!fp)
	{
		/* Silent fail - file may not exist on first run */
		return (-1);
	}

	/* Read and validate header */
	if (fread(&header, sizeof(header), 1, fp) != 1)
	{
		fprintf(stderr, "[FLUID] Error: Failed to read header\n");
		fclose(fp);
		return (-1);
	}

	/* Validate magic */
	if (memcmp(header.magic, FLUID_MAGIC, 4) != 0)
	{
		fprintf(stderr, "[FLUID] Error: Invalid magic (not a .fluid file)\n");
		fclose(fp);
		return (-1);
	}

	/* Validate version */
	if (header.version != FLUID_VERSION)
	{
		fprintf(stderr, "[FLUID] Error: Version mismatch (file=%u, expected=%u)\n",
			header.version, FLUID_VERSION);
		fclose(fp);
		return (-1);
	}

	/* Validate model dimension (CRITICAL for safety) */
	if (header.model_dim != (uint32_t)t->config.dim)
	{
		fprintf(stderr, "[FLUID] Error: Dim mismatch (file=%u, model=%d)\n",
			header.model_dim, t->config.dim);
		fprintf(stderr, "[FLUID] This brain was trained for a different model!\n");
		fclose(fp);
		return (-1);
	}

	/* Load context bias entries */
	loaded = 0;
	if (t->context_bias.keys)
	{
		i = 0;
		while (i < header.n_bias_entries)
		{
			if (fread(&entry, sizeof(entry), 1, fp) != 1)
			{
				fprintf(stderr, "[FLUID] Error: Failed to read bias entry %u\n", i);
				fclose(fp);
				return (-1);
			}

			/* Insert into hashmap using same hash as inference.c */
			h = entry.key;
			h ^= h >> 33;
			h *= 0xff51afd7ed558ccdULL;
			h ^= h >> 33;
			h *= 0xc4ceb9fe1a85ec53ULL;
			h ^= h >> 33;
			idx = (uint32_t)(h % t->context_bias.size);

			/* Linear probing to find slot */
			for (j = 0; j < 16; j++)
			{
				uint32_t cur = (idx + j) % t->context_bias.size;
				if (t->context_bias.keys[cur] == 0 || 
					t->context_bias.keys[cur] == entry.key)
				{
					t->context_bias.keys[cur] = entry.key;
					t->context_bias.tokens[cur] = entry.target_token;
					t->context_bias.biases[cur] = entry.bias;
					if (t->context_bias.keys[cur] == 0)
						t->context_bias.count++;
					loaded++;
					break;
				}
			}
			i++;
		}
	}

	/* Load final adapter if present */
	if (header.has_adapter)
	{
		if (t->final_adapter && t->final_adapter->data)
		{
			size_t adapter_size = (size_t)t->config.dim * t->config.dim * sizeof(t_bf16);
			if (fread(t->final_adapter->data, adapter_size, 1, fp) != 1)
			{
				fprintf(stderr, "[FLUID] Error: Failed to read adapter\n");
				fclose(fp);
				return (-1);
			}
			printf("[FLUID] Loaded final adapter [%d x %d]\n",
				t->config.dim, t->config.dim);
		}
	}

	fclose(fp);

	/* Print timestamp as human-readable */
	char time_buf[64];
	time_t ts = (time_t)header.timestamp;
	struct tm *tm_info = localtime(&ts);
	strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);

	printf("[FLUID] Loaded: %d bias entries from '%s'\n", loaded, path);
	printf("[FLUID] Brain saved at: %s\n", time_buf);
	return (0);
}

void	fluid_print_stats(t_transformer *t)
{
	int	active;

	active = count_active_biases(t);
	printf("[FLUID STATS] Context biases: %d/%d entries used (%.1f%%)\n",
		active, t->context_bias.size,
		t->context_bias.size > 0 ? (float)active / t->context_bias.size * 100 : 0);
	printf("[FLUID STATS] Final adapter: %s\n",
		has_nonzero_adapter(t) ? "has learned weights" : "empty (untrained)");
}
