/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid_io.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 19:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 19:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "fluid_spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
** ===========================================================================
** FLUID I/O LIBRARY
** ===========================================================================
** Robust functions for reading and writing .fluid v2 files.
** Designed to be reusable by:
**   - The inference engine (runtime loading)
**   - The fluid-merge tool
**   - Python bindings (FFI-friendly)
** ===========================================================================
*/

/*
** Create a new header with default values
** @param domain: Knowledge domain (e.g., "coding", "law")
** @param author: Creator identifier (e.g., "DeepSeek-Teacher")
** @param base_hash: XXHash of the base model weights for compatibility
** @return: Initialized header struct
*/
t_fluid_header	fluid_create_header(const char *domain, const char *author,
					uint64_t base_hash)
{
	t_fluid_header	h;

	memset(&h, 0, sizeof(h));
	memcpy(h.magic, FLUID_MAGIC, 4);
	h.version = FLUID_VERSION;
	h.created_at = (uint64_t)time(NULL);
	h.base_model_hash = base_hash;
	h.n_entries = 0;
	h.flags = 0;
	if (domain)
		strncpy(h.domain, domain, FLUID_META_STR_LEN - 1);
	if (author)
		strncpy(h.author, author, FLUID_META_STR_LEN - 1);
	return (h);
}

/*
** Set the description field of a header
** @param h: Header to modify
** @param desc: Description string
*/
void	fluid_set_description(t_fluid_header *h, const char *desc)
{
	if (h && desc)
		strncpy(h->description, desc, FLUID_DESC_LEN - 1);
}

/*
** Write a complete .fluid file
** @param filename: Output path
** @param header: File header (n_entries must be set)
** @param entries: Array of knowledge entries
** @return: FLUID_OK on success, FLUID_ERR_FILE on failure
*/
int	fluid_write_file(const char *filename, t_fluid_header *header,
			t_fluid_entry *entries)
{
	FILE	*f;
	size_t	written;

	f = fopen(filename, "wb");
	if (!f)
		return (FLUID_ERR_FILE);
	/* Write header (512 bytes) */
	written = fwrite(header, sizeof(t_fluid_header), 1, f);
	if (written != 1)
	{
		fclose(f);
		return (FLUID_ERR_FILE);
	}
	/* Write entries (bulk write for performance) */
	if (header->n_entries > 0 && entries)
	{
		written = fwrite(entries, sizeof(t_fluid_entry), header->n_entries, f);
		if (written != header->n_entries)
		{
			fclose(f);
			return (FLUID_ERR_FILE);
		}
	}
	fclose(f);
	return (FLUID_OK);
}

/*
** Read only the header from a .fluid file (for inspection)
** @param filename: Input path
** @param out_header: Output buffer for header
** @return: FLUID_OK, FLUID_ERR_FILE, or FLUID_ERR_MAGIC
*/
int	fluid_read_header(const char *filename, t_fluid_header *out_header)
{
	FILE	*f;

	if (!out_header)
		return (FLUID_ERR_FILE);
	f = fopen(filename, "rb");
	if (!f)
		return (FLUID_ERR_FILE);
	if (fread(out_header, sizeof(t_fluid_header), 1, f) != 1)
	{
		fclose(f);
		return (FLUID_ERR_FILE);
	}
	fclose(f);
	/* Validate magic bytes */
	if (memcmp(out_header->magic, FLUID_MAGIC, 4) != 0)
		return (FLUID_ERR_MAGIC);
	/* Validate version */
	if (out_header->version > FLUID_VERSION)
		return (FLUID_ERR_VERSION);
	return (FLUID_OK);
}

/*
** Read a complete .fluid file (header + entries)
** @param filename: Input path
** @param out_header: Output buffer for header
** @param out_entries: Output pointer for allocated entries array (caller must free)
** @return: FLUID_OK on success, error code on failure
*/
int	fluid_read_file(const char *filename, t_fluid_header *out_header,
			t_fluid_entry **out_entries)
{
	FILE	*f;
	int		ret;

	if (!out_header || !out_entries)
		return (FLUID_ERR_FILE);
	*out_entries = NULL;
	/* Read header first */
	ret = fluid_read_header(filename, out_header);
	if (ret != FLUID_OK)
		return (ret);
	/* Allocate and read entries */
	if (out_header->n_entries == 0)
		return (FLUID_OK);
	*out_entries = malloc(out_header->n_entries * sizeof(t_fluid_entry));
	if (!*out_entries)
		return (FLUID_ERR_FILE);
	f = fopen(filename, "rb");
	if (!f)
	{
		free(*out_entries);
		*out_entries = NULL;
		return (FLUID_ERR_FILE);
	}
	/* Skip header */
	fseek(f, sizeof(t_fluid_header), SEEK_SET);
	/* Read entries */
	if (fread(*out_entries, sizeof(t_fluid_entry), out_header->n_entries, f)
		!= out_header->n_entries)
	{
		free(*out_entries);
		*out_entries = NULL;
		fclose(f);
		return (FLUID_ERR_CORRUPT);
	}
	fclose(f);
	return (FLUID_OK);
}

/*
** Validate compatibility between a .fluid file and current model
** @param header: Header from .fluid file
** @param current_model_hash: XXHash of currently loaded model
** @return: FLUID_OK if compatible, FLUID_ERR_MODEL_HASH if mismatch
*/
int	fluid_validate_compatibility(const t_fluid_header *header,
			uint64_t current_model_hash)
{
	/* Hash of 0 means "ignore compatibility check" */
	if (header->base_model_hash == 0 || current_model_hash == 0)
		return (FLUID_OK);
	if (header->base_model_hash != current_model_hash)
		return (FLUID_ERR_MODEL_HASH);
	return (FLUID_OK);
}

/*
** Get human-readable error message
** @param err: Error code
** @return: Static error string
*/
const char	*fluid_strerror(int err)
{
	if (err == FLUID_OK)
		return ("Success");
	if (err == FLUID_ERR_FILE)
		return ("File I/O error");
	if (err == FLUID_ERR_MAGIC)
		return ("Invalid file format (not FLv2)");
	if (err == FLUID_ERR_VERSION)
		return ("Unsupported version");
	if (err == FLUID_ERR_MODEL_HASH)
		return ("Base model mismatch");
	if (err == FLUID_ERR_CORRUPT)
		return ("File corrupted");
	return ("Unknown error");
}
