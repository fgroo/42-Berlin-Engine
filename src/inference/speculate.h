/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   speculate.h                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/22 08:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/22 08:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef SPECULATE_H
# define SPECULATE_H

# include "inference.h"
# include "bridge.h"

/*
** ============================================================================
** MULTI-TOKEN PREDICTION (MTP) ENGINE
** ============================================================================
** Speculative decoding with heterogeneous models.
** Draft model (Gemma) proposes, Target model (Ministral) verifies.
**
** Key insight: Draft model is cheap (270M), Target is expensive (3B).
** If Draft guesses correctly 60% of the time, we get ~2.5x speedup.
** ============================================================================
*/

# define MTP_MAX_DRAFT    16   /* Maximum lookahead tokens */
# define MTP_DEFAULT_DRAFT 5   /* Default lookahead (tune per model) */

typedef struct s_mtp_engine
{
	t_transformer	*target;         /* Main model (e.g., Ministral 3B) */
	t_transformer	*draft;          /* Draft model (e.g., Gemma 270M) */
	t_tokenizer		*target_tok;     /* Target tokenizer */
	t_tokenizer		*draft_tok;      /* Draft tokenizer */
	t_token_bridge	bridge;          /* Cross-tokenizer translation */
	int				n_draft;         /* Number of tokens to draft */
	int				is_speculative;  /* 1 = MTP enabled, 0 = fallback */
	/* Stats */
	int				total_drafted;   /* Total tokens drafted */
	int				total_accepted;  /* Total tokens accepted */
	int				total_rejected;  /* Total tokens rejected */
}	t_mtp_engine;

/*
** Initialize MTP engine with target and draft models.
** Returns 0 on success, -1 on failure.
*/
int		mtp_init(t_mtp_engine *eng, t_transformer *target, t_transformer *draft,
			t_tokenizer *target_tok, t_tokenizer *draft_tok);

/*
** Generate tokens using speculative decoding (BURST MODE).
** Fills out_tokens buffer with all accepted/generated tokens.
** Returns the number of tokens written to out_tokens (1 to n_draft+1).
**
** @param eng          MTP engine context
** @param prompt_token Last token (target ID) - the input to continue from
** @param pos          Current sequence position
** @param out_tokens   Buffer to receive generated tokens (must be >= MTP_MAX_DRAFT+1)
** @return             Number of tokens written to out_tokens
*/
int		mtp_generate(t_mtp_engine *eng, int prompt_token, int pos,
			int *out_tokens);

/*
** Print MTP statistics.
*/
void	mtp_stats(t_mtp_engine *eng);

/*
** Free MTP engine resources.
*/
void	mtp_free(t_mtp_engine *eng);

/*
** ============================================================================
** KV CACHE UTILITIES
** ============================================================================
*/

/*
** Rewind KV cache to specified position.
** Invalidates all cached states beyond new_pos.
** Used after speculation rejection.
*/
void	kv_cache_rewind(t_transformer *t, int new_pos);

#endif
