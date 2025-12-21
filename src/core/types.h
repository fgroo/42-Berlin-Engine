/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   types.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/21 16:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/21 16:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPES_H
# define TYPES_H

# include <stdint.h>
# include <stddef.h>

/*
** ============================================================================
** MOPD (Multi-Teacher On-Policy Distillation) Types
** ============================================================================
** These structures define the contract between forge.py (Python) and the
** C engine for knowledge distillation from larger teacher models.
** ============================================================================
*/

/*
** Sparse Probability Entry
** We store probabilities (0.0 - 1.0), NOT logprobs!
** Conversion from logprob -> prob happens in the API handler.
**
** Teachers like GPT-4 return top-K logprobs, not full vocab distribution.
** Sending 128K floats per token is insane - we send ~20 sparse entries.
*/
typedef struct s_sparse_prob
{
	int		token_id;	/* Vocab index */
	float	prob;		/* P(x), range [0.0, 1.0] */
}	t_sparse_prob;

/*
** Distillation Context for the Backward Pass
** Populated by server.c, passed to backward.c
*/
typedef struct s_distill_request
{
	t_sparse_prob	*teacher_probs;	/* Array of Top-K Teacher Probs */
	int				num_probs;		/* Size of array (e.g., 20) */
	int				target_token;	/* Ground truth next token */
	float			alpha;			/* 0.0 = Hard Labels, 1.0 = Teacher Only */
}	t_distill_request;

#endif /* TYPES_H */
