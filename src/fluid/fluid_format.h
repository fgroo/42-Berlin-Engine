/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid_format.h                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/24 02:10:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/24 02:10:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FLUID_FORMAT_H
# define FLUID_FORMAT_H

# include <stdint.h>

/*
** ============================================================================
** .FLUID FILE FORMAT - "Skill Cartridges"
** ============================================================================
** Binary format for storing learned adapter weights.
** Can be loaded instantly to inject knowledge into the model.
** ============================================================================
*/

# define FLUID_MAGIC    0x44495546  /* "FUID" in hex (little endian) */
# define FLUID_VERSION  1

/*
** File header - must be at start of every .fluid file
*/
typedef struct s_fluid_header
{
	uint32_t	magic;           /* Must be FLUID_MAGIC */
	uint32_t	version;         /* Format version */
	char		base_model[64];  /* Model name for compatibility check */
	
	/* Topology */
	int			dim;             /* Model hidden dimension */
	int			n_layers;        /* Total layers */
	int			n_trainable;     /* Number of trainable layer adapters */
	int			vocab_size;      /* Vocabulary size (for logit_bias) */
	
	/* Stats (for UI/debugging) */
	float		trained_tokens;  /* Total tokens trained on */
	float		final_loss;      /* Last recorded loss */
	uint64_t	timestamp;       /* Unix timestamp of save */
	
	/* Reserved for future use */
	uint64_t	reserved[4];
}	t_fluid_header;

/*
** File Layout after header:
** 
** [HEADER]                        sizeof(t_fluid_header)
** [FINAL_ADAPTER_WEIGHTS]         dim * dim * sizeof(float)
** [FINAL_ADAPTER_GRAD]            dim * dim * sizeof(float)  (optional)
** [LOGIT_BIAS]                    vocab_size * sizeof(float)
** [CONTEXT_BIAS]                  65536 * sizeof(float)
** [LAYER_0_W2_WEIGHT]             (if trainable)
** [LAYER_0_W2_GRAD]               (if trainable)
** ...
** [LAYER_N_W2_WEIGHT]             (if trainable)
** [LAYER_N_W2_GRAD]               (if trainable)
*/

#endif
