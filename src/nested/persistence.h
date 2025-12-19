/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   persistence.h                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 10:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 19:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef PERSISTENCE_H
# define PERSISTENCE_H

# include <stdint.h>
# include "../fluid/fluid_spec.h"

/* Forward declaration - avoid circular include */
struct s_transformer;

/*
** ============================================================================
** FLUID PERSISTENCE - Protocol v2 Integration
** ============================================================================
** This module bridges the inference engine with the Fluid Protocol v2.
** It uses the t_fluid_entry format from fluid_spec.h for portable storage.
**
** Features:
**   - Sparse storage (only non-zero entries)
**   - Metadata support (domain, author, description)
**   - Base model hash for compatibility checking
**   - Support for final adapter weights
** ============================================================================
*/

/*
** Legacy format support - for loading old v1 files
*/
# define FLUID_V1_MAGIC "42FL"

typedef struct s_fluid_v1_header
{
	char		magic[4];		/* "42FL" */
	uint32_t	version;		/* Format version (1) */
	uint32_t	model_dim;		/* For validation (must match current model) */
	uint64_t	timestamp;		/* Unix epoch when saved */
	uint32_t	n_bias_entries;	/* Number of context bias entries */
	uint32_t	has_adapter;	/* 1 if final_adapter is stored */
	uint32_t	reserved[2];	/* Future expansion */
}	t_fluid_v1_header;

typedef struct s_fluid_v1_bias_entry
{
	uint64_t	key;			/* Hash key: (prev_token << 32) | current_token */
	int32_t		target_token;	/* Token ID to bias */
	float		bias;			/* Bias value */
}	t_fluid_v1_bias_entry;

/*
** v2 save options (passed to fluid_save_v2)
*/
typedef struct s_fluid_save_opts
{
	const char	*domain;		/* e.g., "coding", "law" */
	const char	*author;		/* e.g., "42-Berlin-Engine" */
	const char	*description;	/* Human-readable description */
	uint64_t	base_model_hash;	/* XXHash of base model (0 = skip check) */
}	t_fluid_save_opts;

/*
** Save the current fluid state (learned weights + biases) to a file.
** Uses Fluid Protocol v2 format with full metadata.
**
** @param t: Transformer with learned state
** @param path: Output file path (e.g., "brain.fluid")
** @param opts: Optional metadata (can be NULL for defaults)
** @return: 0 on success, -1 on error
*/
int		fluid_save_v2(struct s_transformer *t, const char *path,
			const t_fluid_save_opts *opts);

/*
** Legacy save function (writes v2 with default metadata)
*/
int		fluid_save(struct s_transformer *t, const char *path);

/*
** Load fluid state from a file and merge into current model.
** Auto-detects v1 vs v2 format.
**
** @param t: Transformer to load state into
** @param path: Input file path
** @return: 0 on success, -1 on error (file not found, dim mismatch, etc.)
*/
int		fluid_load(struct s_transformer *t, const char *path);

/*
** Print summary of stored fluid state (for debugging)
*/
void	fluid_print_stats(struct s_transformer *t);

#endif
