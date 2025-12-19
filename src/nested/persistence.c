/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   persistence.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 10:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 19:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "persistence.h"
#include "../fluid/fluid_io.h"
#include "inference/inference.h"
#include "config.h"  /* LINEAR_PROBE_LIMIT */
#include "tensor/tensor.h"  /* t_bf16 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
** ============================================================================
** FLUID STATE PERSISTENCE - Protocol v2 Implementation
** ============================================================================
** "Docker made software portable. .fluid files make skills portable."
**
** This module saves learned knowledge as portable capsules that can be:
**   - Inspected with fluid-info
**   - Merged with fluid-merge
**   - Shared across compatible model instances
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

/*
** Save fluid state using Fluid Protocol v2 format
** Includes full metadata for the Hive Mind ecosystem
*/
int	fluid_save_v2(t_transformer *t, const char *path,
		const t_fluid_save_opts *opts)
{
	FILE			*fp;
	t_fluid_header	header;
	t_fluid_entry	entry;
	int				i;
	int				written;
	size_t			adapter_size;

	/* Create v2 header */
	header = fluid_create_header(
		opts ? opts->domain : "general",
		opts ? opts->author : "42-Berlin-Engine",
		opts ? opts->base_model_hash : 0
	);
	if (opts && opts->description)
		fluid_set_description(&header, opts->description);
	header.n_entries = count_active_biases(t);
	if (has_nonzero_adapter(t))
		header.flags |= FLUID_FLAG_DELTA;  /* Mark as having weight deltas */

	fp = fopen(path, "wb");
	if (!fp)
	{
		fprintf(stderr, "[FLUID] Error: Cannot open '%s' for writing\n", path);
		return (-1);
	}

	/* Write v2 header (512 bytes) */
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
				entry.context_hash = t->context_bias.keys[i];
				entry.target_token = t->context_bias.tokens[i];
				entry.weight = t->context_bias.biases[i];
				if (fwrite(&entry, sizeof(entry), 1, fp) != 1)
				{
					fprintf(stderr, "[FLUID] Error: Failed to write entry\n");
					fclose(fp);
					return (-1);
				}
				written++;
			}
			i++;
		}
	}

	/* Write final adapter if present (appended after entries) */
	if (has_nonzero_adapter(t))
	{
		adapter_size = (size_t)t->config.dim * t->config.dim * sizeof(t_bf16);
		if (fwrite(t->final_adapter->data, adapter_size, 1, fp) != 1)
		{
			fprintf(stderr, "[FLUID] Error: Failed to write adapter\n");
			fclose(fp);
			return (-1);
		}
	}

	fclose(fp);
	printf("[FLUID v2] Saved: %d patterns, adapter=%s\n",
		written, has_nonzero_adapter(t) ? "yes" : "no");
	printf("[FLUID v2] Domain: %s, Author: %s\n",
		header.domain[0] ? header.domain : "(none)",
		header.author[0] ? header.author : "(none)");
	return (0);
}

/*
** Legacy save function - wraps v2 with default metadata
*/
int	fluid_save(t_transformer *t, const char *path)
{
	t_fluid_save_opts	opts;

	memset(&opts, 0, sizeof(opts));
	opts.domain = "general";
	opts.author = "42-Berlin-Engine";
	opts.description = "Learned knowledge from interactive session";
	return (fluid_save_v2(t, path, &opts));
}

/*
** Load legacy v1 format
*/
static int	fluid_load_v1(t_transformer *t, FILE *fp, t_fluid_v1_header *h)
{
	t_fluid_v1_bias_entry	entry;
	uint32_t				i;
	int						loaded;
	uint64_t				hash;
	uint32_t				idx;
	int						j;

	/* Validate model dimension */
	if (h->model_dim != (uint32_t)t->config.dim)
	{
		fprintf(stderr, "[FLUID v1] Dim mismatch (file=%u, model=%d)\n",
			h->model_dim, t->config.dim);
		return (-1);
	}

	/* Load context bias entries */
	loaded = 0;
	if (t->context_bias.keys)
	{
		i = 0;
		while (i < h->n_bias_entries)
		{
			if (fread(&entry, sizeof(entry), 1, fp) != 1)
			{
				fprintf(stderr, "[FLUID v1] Failed to read entry %u\n", i);
				return (-1);
			}
			/* Hash and insert */
			hash = entry.key;
			hash ^= hash >> 33;
			hash *= 0xff51afd7ed558ccdULL;
			hash ^= hash >> 33;
			hash *= 0xc4ceb9fe1a85ec53ULL;
			hash ^= hash >> 33;
			idx = (uint32_t)(hash % t->context_bias.size);
			for (j = 0; j < LINEAR_PROBE_LIMIT; j++)
			{
				uint32_t cur = (idx + j) % t->context_bias.size;
				if (t->context_bias.keys[cur] == 0 ||
					t->context_bias.keys[cur] == entry.key)
				{
					if (t->context_bias.keys[cur] == 0)
						t->context_bias.count++;
					t->context_bias.keys[cur] = entry.key;
					t->context_bias.tokens[cur] = entry.target_token;
					t->context_bias.biases[cur] = entry.bias;
					loaded++;
					break;
				}
			}
			i++;
		}
	}

	/* Load adapter if present */
	if (h->has_adapter && t->final_adapter && t->final_adapter->data)
	{
		size_t adapter_size = (size_t)t->config.dim * t->config.dim * sizeof(t_bf16);
		if (fread(t->final_adapter->data, adapter_size, 1, fp) != 1)
		{
			fprintf(stderr, "[FLUID v1] Failed to read adapter\n");
			return (-1);
		}
	}

	printf("[FLUID v1] Loaded: %d patterns (legacy format)\n", loaded);
	return (0);
}

/*
** Load Fluid Protocol v2 format
*/
static int	fluid_load_v2(t_transformer *t, FILE *fp, t_fluid_header *h)
{
	t_fluid_entry	entry;
	uint32_t		i;
	int				loaded;
	uint64_t		hash;
	uint32_t		idx;
	int				j;

	/* Load context bias entries */
	loaded = 0;
	if (t->context_bias.keys)
	{
		i = 0;
		while (i < h->n_entries)
		{
			if (fread(&entry, sizeof(entry), 1, fp) != 1)
			{
				fprintf(stderr, "[FLUID v2] Failed to read entry %u\n", i);
				return (-1);
			}
			/* Hash and insert */
			hash = entry.context_hash;
			hash ^= hash >> 33;
			hash *= 0xff51afd7ed558ccdULL;
			hash ^= hash >> 33;
			hash *= 0xc4ceb9fe1a85ec53ULL;
			hash ^= hash >> 33;
			idx = (uint32_t)(hash % t->context_bias.size);
			for (j = 0; j < LINEAR_PROBE_LIMIT; j++)
			{
				uint32_t cur = (idx + j) % t->context_bias.size;
				if (t->context_bias.keys[cur] == 0 ||
					t->context_bias.keys[cur] == entry.context_hash)
				{
					if (t->context_bias.keys[cur] == 0)
						t->context_bias.count++;
					t->context_bias.keys[cur] = entry.context_hash;
					t->context_bias.tokens[cur] = entry.target_token;
					t->context_bias.biases[cur] = entry.weight;
					loaded++;
					break;
				}
			}
			i++;
		}
	}

	/* Load adapter if present (DELTA flag) */
	if ((h->flags & FLUID_FLAG_DELTA) &&
		t->final_adapter && t->final_adapter->data)
	{
		size_t adapter_size = (size_t)t->config.dim * t->config.dim * sizeof(t_bf16);
		if (fread(t->final_adapter->data, adapter_size, 1, fp) != 1)
		{
			fprintf(stderr, "[FLUID v2] Failed to read adapter\n");
			return (-1);
		}
		printf("[FLUID v2] Loaded final adapter [%d x %d]\n",
			t->config.dim, t->config.dim);
	}

	/* Print metadata */
	printf("[FLUID v2] Loaded: %d patterns\n", loaded);
	if (h->domain[0])
		printf("[FLUID v2] Domain: %s\n", h->domain);
	if (h->author[0])
		printf("[FLUID v2] Author: %s\n", h->author);
	return (0);
}

/*
** Load fluid state - auto-detects v1 vs v2 format
*/
int	fluid_load(t_transformer *t, const char *path)
{
	FILE			*fp;
	char			magic[4];
	int				ret;

	fp = fopen(path, "rb");
	if (!fp)
		return (-1);  /* Silent fail - file may not exist */

	/* Read magic to detect version */
	if (fread(magic, 4, 1, fp) != 1)
	{
		fclose(fp);
		return (-1);
	}

	/* Detect format by magic bytes */
	if (memcmp(magic, FLUID_MAGIC, 4) == 0)
	{
		/* v2 format - rewind and read full header */
		t_fluid_header h;
		fseek(fp, 0, SEEK_SET);
		if (fread(&h, sizeof(h), 1, fp) != 1)
		{
			fclose(fp);
			return (-1);
		}
		ret = fluid_load_v2(t, fp, &h);
		fclose(fp);
		return (ret);
	}
	else if (memcmp(magic, FLUID_V1_MAGIC, 4) == 0)
	{
		/* v1 format - legacy support */
		t_fluid_v1_header h;
		fseek(fp, 0, SEEK_SET);
		if (fread(&h, sizeof(h), 1, fp) != 1)
		{
			fclose(fp);
			return (-1);
		}
		ret = fluid_load_v1(t, fp, &h);
		fclose(fp);
		return (ret);
	}
	else
	{
		fprintf(stderr, "[FLUID] Unknown format: '%c%c%c%c'\n",
			magic[0], magic[1], magic[2], magic[3]);
		fclose(fp);
		return (-1);
	}
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
