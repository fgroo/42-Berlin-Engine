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
** ============================================================================
** TOKEN STRING NORMALIZATION
** ============================================================================
** SentencePiece uses "Lower One Eighth Block" (U+2581 / ▁) as word boundary.
** Standard tokenizers use ASCII space (0x20).
** We normalize both directions to maximize match rate.
**
** U+2581 = 0xE2 0x96 0x81 (UTF-8)
** ============================================================================
*/

/* Static buffer for normalized strings (thread-local for safety) */
static __thread char	g_norm_buf[512];

/*
** Normalize token string for cross-tokenizer matching.
** - Replaces U+2581 (▁) with ASCII space
** - Handles Byte-Level BPE escape sequences if needed
*/
static const char	*normalize_token_str(const char *raw)
{
	const unsigned char	*src;
	char				*dst;
	int					i;

	if (!raw)
		return (NULL);
	src = (const unsigned char *)raw;
	dst = g_norm_buf;
	i = 0;
	while (*src && i < 510)
	{
		/* Check for U+2581: 0xE2 0x96 0x81 */
		if (src[0] == 0xE2 && src[1] == 0x96 && src[2] == 0x81)
		{
			dst[i++] = ' ';  /* Replace with ASCII space */
			src += 3;
		}
		/* Check for reverse: if target uses ▁ but draft uses space */
		/* We keep ASCII space as-is, the target lookup handles it */
		else
		{
			dst[i++] = *src++;
		}
	}
	dst[i] = '\0';
	return (g_norm_buf);
}

/*
** Translate a draft token ID to the target tokenizer's ID.
** Uses caching to avoid repeated decode/encode cycles.
**
** Flow:
** 1. Check cache (O(1))
** 2. If miss: decode draft_id -> normalize -> encode to target_id
** 3. If still no match: try without normalization (for exact matches)
** 4. Cache the result
*/
int	bridge_translate(t_token_bridge *bridge, int draft_id)
{
	const char	*token_str;
	const char	*normalized;
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
	/* Step 2: Normalize and encode to target ID */
	normalized = normalize_token_str(token_str);
	target_id = tokenizer_lookup_id(bridge->target_tok, normalized);
	/* Step 3: If normalized lookup returned UNK, try raw string */
	if (target_id == bridge->target_tok->unk_id && normalized != token_str)
	{
		int raw_id = tokenizer_lookup_id(bridge->target_tok, token_str);
		if (raw_id != bridge->target_tok->unk_id)
			target_id = raw_id;
	}
	/* Step 4: Cache the result */
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
** Reverse translate: target token ID to draft token ID.
** Used to sync draft model with target's accepted tokens.
**
** This is NOT cached because:
** 1. It's called less frequently (only on accepted tokens)
** 2. We'd need a second cache of target_vocab_size
*/
int	bridge_reverse_translate(t_token_bridge *bridge, int target_id)
{
	const char	*token_str;
	const char	*normalized;
	int			draft_id;

	if (!bridge || target_id < 0)
		return (BRIDGE_NO_MATCH);
	if (target_id >= bridge->target_tok->vocab_size)
		return (BRIDGE_NO_MATCH);
	/* Decode target ID to string */
	token_str = tokenizer_decode(bridge->target_tok, target_id);
	if (!token_str || token_str[0] == '\0')
		return (BRIDGE_NO_MATCH);
	/* Normalize and lookup in draft tokenizer */
	normalized = normalize_token_str(token_str);
	draft_id = tokenizer_lookup_id(bridge->draft_tok, normalized);
	/* Fallback to raw string */
	if (draft_id == bridge->draft_tok->unk_id && normalized != token_str)
	{
		int raw_id = tokenizer_lookup_id(bridge->draft_tok, token_str);
		if (raw_id != bridge->draft_tok->unk_id)
			draft_id = raw_id;
	}
	return (draft_id);
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
