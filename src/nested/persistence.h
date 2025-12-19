/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   persistence.h                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 10:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 10:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef PERSISTENCE_H
# define PERSISTENCE_H

# include <stdint.h>

/* Forward declaration - avoid circular include */
struct s_transformer;

/*
** ============================================================================
** FLUID FILE FORMAT (.fluid)
** ============================================================================
** Binary format for persistent learned state.
** Only non-zero entries are stored (sparse storage).
**
** Layout:
**   [Header: 32 bytes]
**   [Context Bias Entries: 16 bytes each]
**   [Final Adapter Flag: 4 bytes]
**   [Final Adapter Data: dim*dim*2 bytes if flag=1]
** ============================================================================
*/

# define FLUID_MAGIC "42FL"
# define FLUID_VERSION 1

/*
** File header - 32 bytes
*/
typedef struct s_fluid_header
{
	char		magic[4];		/* "42FL" */
	uint32_t	version;		/* Format version (1) */
	uint32_t	model_dim;		/* For validation (must match current model) */
	uint64_t	timestamp;		/* Unix epoch when saved */
	uint32_t	n_bias_entries;	/* Number of context bias entries */
	uint32_t	has_adapter;	/* 1 if final_adapter is stored */
	uint32_t	reserved[2];	/* Future expansion */
}	t_fluid_header;

/*
** Context bias entry - 16 bytes (sparse storage)
*/
typedef struct s_fluid_bias_entry
{
	uint64_t	key;			/* Hash key: (prev_token << 32) | current_token */
	int32_t		target_token;	/* Token ID to bias */
	float		bias;			/* Bias value */
}	t_fluid_bias_entry;

/*
** Save the current fluid state (learned weights + biases) to a file.
** Only non-zero entries are saved (sparse format).
**
** @param t: Transformer with learned state
** @param path: Output file path (e.g., "brain.fluid")
** @return: 0 on success, -1 on error
*/
int		fluid_save(struct s_transformer *t, const char *path);

/*
** Load fluid state from a file and merge into current model.
** Validates model dimensions before loading.
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
