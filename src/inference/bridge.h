/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bridge.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/22 07:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/22 07:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef BRIDGE_H
# define BRIDGE_H

# include "tokenizer/tokenizer.h"

/*
** ============================================================================
** MTP UNIVERSAL BRIDGE
** ============================================================================
** Cross-tokenizer translation layer for Multi-Token Prediction.
** Enables heterogeneous model setups (e.g., Gemma draft + Ministral target).
**
** The bridge caches translations to avoid repeated tokenize/detokenize cycles.
** Cache hit: O(1) array lookup
** Cache miss: O(tokenizer_decode + tokenizer_lookup_id)
** ============================================================================
*/

# define BRIDGE_CACHE_INVALID -2  /* Not yet translated */
# define BRIDGE_NO_MATCH      -1  /* Translated but no match in target vocab */

typedef struct s_token_bridge
{
	t_tokenizer	*draft_tok;      /* Draft model tokenizer (e.g., Gemma) */
	t_tokenizer	*target_tok;     /* Target model tokenizer (e.g., Ministral) */
	int			*cache;          /* Translation cache: draft_id -> target_id */
	int			cache_size;      /* Size of cache (= draft vocab size) */
	int			cache_hits;      /* Stats: cache hit count */
	int			cache_misses;    /* Stats: cache miss count */
}	t_token_bridge;

/*
** Initialize the token bridge.
** Allocates cache for draft vocab size.
*/
int		bridge_init(t_token_bridge *bridge, t_tokenizer *draft_tok,
			t_tokenizer *target_tok);

/*
** Translate draft token ID to target token ID.
** Returns target_id or BRIDGE_NO_MATCH if no mapping exists.
*/
int		bridge_translate(t_token_bridge *bridge, int draft_id);

/*
** Batch translation for speculative decoding.
** Translates n draft tokens to target IDs.
*/
void	bridge_translate_batch(t_token_bridge *bridge, const int *draft_ids,
			int *target_ids, int n);

/*
** Print bridge statistics.
*/
void	bridge_stats(t_token_bridge *bridge);

/*
** Reverse translate: target token ID to draft token ID.
** Used to sync draft model with target's accepted tokens.
** Note: This is NOT cached (less frequent operation).
*/
int		bridge_reverse_translate(t_token_bridge *bridge, int target_id);

/*
** Free bridge resources.
*/
void	bridge_free(t_token_bridge *bridge);

#endif
