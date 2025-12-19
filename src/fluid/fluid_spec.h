/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid_spec.h                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 19:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 19:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FLUID_SPEC_H
# define FLUID_SPEC_H

# include <stdint.h>

/*
** ===========================================================================
** FLUID PROTOCOL v2 - Portable Knowledge Capsules
** ===========================================================================
** "Docker made software portable. .fluid files make skills portable."
**
** This header defines the binary format for storing trained knowledge
** that can be:
**   - Inspected (fluid-info tool)
**   - Merged (fluid-merge tool)  
**   - Hot-swapped at runtime
**
** The format is designed for sparse storage: only learned patterns are
** saved, not the entire weight delta.
** ===========================================================================
*/

# define FLUID_MAGIC "FLv2"
# define FLUID_VERSION 2

/* String limits for metadata */
# define FLUID_META_STR_LEN 64
# define FLUID_DESC_LEN 256

/* Maximum entries in a single file (256K patterns) */
# define FLUID_MAX_ENTRIES 262144

/*
** File Header - First 512 bytes of every .fluid file
** Padded for alignment and future extensibility
*/
typedef struct s_fluid_header
{
	char		magic[4];              /* Must be "FLv2" */
	uint32_t	version;               /* Format version (currently 2) */
	uint64_t	created_at;            /* Unix timestamp */
	uint64_t	base_model_hash;       /* XXHash of base model for compatibility check */

	/* Metadata for discovery/filtering */
	char		domain[FLUID_META_STR_LEN];       /* e.g. "coding", "law", "medical" */
	char		author[FLUID_META_STR_LEN];       /* e.g. "DeepSeek-Teacher" */
	char		description[FLUID_DESC_LEN];     /* Human-readable description */

	/* Entry count and flags */
	uint32_t	n_entries;             /* Number of learned patterns */
	uint32_t	flags;                 /* Reserved (compression, encryption) */

	/* Padding to 512 bytes for future expansion */
	uint8_t		reserved[512 - 4 - 4 - 8 - 8 - 64 - 64 - 256 - 4 - 4];
}	t_fluid_header;

/*
** Knowledge Entry - A single learned pattern (16 bytes)
** Represents: "When context_hash is seen, boost target_token by weight"
*/
typedef struct s_fluid_entry
{
	uint64_t	context_hash;          /* Trigger: bigram/context hash */
	int32_t		target_token;          /* Response: token ID to boost */
	float		weight;                /* Strength: how much to boost */
}	t_fluid_entry;

/*
** Flag bits for t_fluid_header.flags
*/
# define FLUID_FLAG_COMPRESSED  0x0001  /* Entries are zstd compressed */
# define FLUID_FLAG_ENCRYPTED   0x0002  /* Entries are encrypted */
# define FLUID_FLAG_DELTA       0x0004  /* Weights are delta from base */
# define FLUID_FLAG_VERIFIED    0x0008  /* Trained with teacher verification */

/*
** Error codes for fluid_* functions
*/
# define FLUID_OK               0
# define FLUID_ERR_FILE         -1  /* File I/O error */
# define FLUID_ERR_MAGIC        -2  /* Invalid magic bytes */
# define FLUID_ERR_VERSION      -3  /* Unsupported version */
# define FLUID_ERR_MODEL_HASH   -4  /* Base model mismatch */
# define FLUID_ERR_CORRUPT      -5  /* Data corruption detected */

#endif
