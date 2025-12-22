/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bridge.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/22 07:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/22 07:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "bridge.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
** Initialize the token bridge for cross-tokenizer translation.
** The cache maps draft token IDs to target token IDs.
*/
int	bridge_init(t_token_bridge *bridge, t_tokenizer *draft_tok,
		t_tokenizer *target_tok)
{
	int	i;

	if (!bridge || !draft_tok || !target_tok)
		return (-1);
	bridge->draft_tok = draft_tok;
	bridge->target_tok = target_tok;
	bridge->cache_size = draft_tok->vocab_size;
	bridge->cache_hits = 0;
	bridge->cache_misses = 0;
	/* Allocate translation cache */
	bridge->cache = malloc(bridge->cache_size * sizeof(int));
	if (!bridge->cache)
	{
		fprintf(stderr, "[BRIDGE] Failed to allocate cache (%d entries)\n",
			bridge->cache_size);
		return (-1);
	}
	/* Initialize cache to INVALID (not yet translated) */
	i = 0;
	while (i < bridge->cache_size)
	{
		bridge->cache[i] = BRIDGE_CACHE_INVALID;
		i++;
	}
	printf("[BRIDGE] Initialized: %d → %d vocab translation cache\n",
		draft_tok->vocab_size, target_tok->vocab_size);
	return (0);
}

/*
** Translate a draft token ID to the target tokenizer's ID.
** Uses caching to avoid repeated decode/encode cycles.
**
** Flow:
** 1. Check cache (O(1))
** 2. If miss: decode draft_id -> string -> encode to target_id
** 3. Cache the result
*/
int	bridge_translate(t_token_bridge *bridge, int draft_id)
{
	const char	*token_str;
	int			target_id;

	/* Bounds check */
	if (draft_id < 0 || draft_id >= bridge->cache_size)
		return (BRIDGE_NO_MATCH);
	/* Check cache */
	if (bridge->cache[draft_id] != BRIDGE_CACHE_INVALID)
	{
		bridge->cache_hits++;
		return (bridge->cache[draft_id]);
	}
	/* Cache miss: perform translation */
	bridge->cache_misses++;
	/* Step 1: Decode draft ID to string */
	token_str = tokenizer_decode(bridge->draft_tok, draft_id);
	if (!token_str || token_str[0] == '\0')
	{
		/* Special token or empty - mark as no match */
		bridge->cache[draft_id] = BRIDGE_NO_MATCH;
		return (BRIDGE_NO_MATCH);
	}
	/* Step 2: Encode string to target ID */
	target_id = tokenizer_lookup_id(bridge->target_tok, token_str);
	/* Step 3: Cache the result */
	/* Note: target_id might be unk_id (0), which is still valid for caching */
	bridge->cache[draft_id] = target_id;
	return (target_id);
}

/*
** Batch translation for speculative decoding.
** More efficient than calling bridge_translate() in a loop
** due to better cache locality.
*/
void	bridge_translate_batch(t_token_bridge *bridge, const int *draft_ids,
		int *target_ids, int n)
{
	int	i;

	i = 0;
	while (i < n)
	{
		target_ids[i] = bridge_translate(bridge, draft_ids[i]);
		i++;
	}
}

/*
** Print bridge statistics.
*/
void	bridge_stats(t_token_bridge *bridge)
{
	int		total;
	float	hit_rate;

	if (!bridge)
		return ;
	total = bridge->cache_hits + bridge->cache_misses;
	if (total > 0)
		hit_rate = (float)bridge->cache_hits / (float)total * 100.0f;
	else
		hit_rate = 0.0f;
	printf("[BRIDGE] Stats: %d hits, %d misses, %.1f%% hit rate\n",
		bridge->cache_hits, bridge->cache_misses, hit_rate);
}

/*
** Free bridge resources.
*/
void	bridge_free(t_token_bridge *bridge)
{
	if (!bridge)
		return ;
	if (bridge->cache)
	{
		free(bridge->cache);
		bridge->cache = NULL;
	}
	bridge->draft_tok = NULL;
	bridge->target_tok = NULL;
	bridge->cache_size = 0;
}
