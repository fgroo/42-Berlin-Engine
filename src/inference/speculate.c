/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   speculate.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/22 08:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/22 08:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "speculate.h"
#include <stdio.h>
#include <string.h>

/*
** ============================================================================
** MTP ENGINE INITIALIZATION
** ============================================================================
*/

int	mtp_init(t_mtp_engine *eng, t_transformer *target, t_transformer *draft,
		t_tokenizer *target_tok, t_tokenizer *draft_tok)
{
	if (!eng || !target)
		return (-1);
	memset(eng, 0, sizeof(t_mtp_engine));
	eng->target = target;
	eng->draft = draft;
	eng->target_tok = target_tok;
	eng->draft_tok = draft_tok;
	eng->n_draft = MTP_DEFAULT_DRAFT;
	/* MTP requires both models and tokenizers */
	if (draft && target_tok && draft_tok)
	{
		if (bridge_init(&eng->bridge, draft_tok, target_tok) == 0)
		{
			eng->is_speculative = 1;
			printf("[MTP] Speculative decoding enabled (draft=%d tokens)\n",
				eng->n_draft);
		}
		else
		{
			printf("[MTP] Bridge init failed, falling back to standard\n");
			eng->is_speculative = 0;
		}
	}
	else
	{
		printf("[MTP] Draft model or tokenizer missing, using standard mode\n");
		eng->is_speculative = 0;
	}
	return (0);
}

/*
** ============================================================================
** KV CACHE REWIND
** ============================================================================
** Invalidates cached states beyond new_pos.
** For dense attention: just update position counter.
** For paged attention: would need to free/invalidate pages.
*/

void	kv_cache_rewind(t_transformer *t, int new_pos)
{
	if (!t)
		return ;
	/* For now, just reset position counter */
	/* The next forward pass will overwrite stale cache entries */
	/* Future: Handle paged KV cache page invalidation */
	(void)new_pos;  /* Suppress unused warning for now */
	/* 
	** Note: t_transformer doesn't have a global 'pos' field.
	** Position is passed per-call to transformer_forward.
	** The KV cache automatically handles position via the pos parameter.
	** So "rewind" for dense attention is essentially a no-op.
	**
	** For paged attention (DeepSeek V3 style), we would:
	** 1. Find all pages allocated beyond new_pos
	** 2. Decrement refcounts
	** 3. Return pages to pool if refcount == 0
	*/
}

/*
** ============================================================================
** SPECULATIVE DECODING CORE (BURST MODE)
** ============================================================================
** The heart of MTP. Draft model proposes, Target model verifies.
** Returns ALL accepted tokens in out_tokens buffer for streaming.
*/

int	mtp_generate(t_mtp_engine *eng, int prompt_token, int pos, int *out_tokens)
{
	int			draft_tokens_gemma[MTP_MAX_DRAFT];
	int			draft_tokens_target[MTP_MAX_DRAFT];
	int			gemma_input;
	int			curr_gemma;
	int			n_out;
	int			check_input;
	int			target_pred;
	int			i;
	int			j;
	float		*logits;
	float		max_val;
	int			max_id;

	/* ====================================================================== */
	/* 0. FALLBACK: Standard mode if MTP disabled                             */
	/* ====================================================================== */
	if (!eng->draft || !eng->is_speculative || !out_tokens)
	{
		logits = transformer_forward(eng->target, prompt_token, pos);
		max_id = 0;
		max_val = logits[0];
		for (i = 1; i < eng->target->config.vocab_size; i++)
		{
			if (logits[i] > max_val)
			{
				max_val = logits[i];
				max_id = i;
			}
		}
		if (out_tokens)
			out_tokens[0] = max_id;
		return (1);
	}

	/* ====================================================================== */
	/* 1. SYNC PHASE                                                          */
	/* ====================================================================== */
	gemma_input = bridge_reverse_translate(&eng->bridge, prompt_token);
	if (gemma_input == BRIDGE_NO_MATCH)
	{
		/* Fallback: Can't translate, use standard path */
		logits = transformer_forward(eng->target, prompt_token, pos);
		max_id = 0;
		max_val = logits[0];
		for (i = 1; i < eng->target->config.vocab_size; i++)
		{
			if (logits[i] > max_val)
			{
				max_val = logits[i];
				max_id = i;
			}
		}
		out_tokens[0] = max_id;
		return (1);
	}

	/* ====================================================================== */
	/* 2. DRAFTING PHASE                                                      */
	/* ====================================================================== */
	curr_gemma = gemma_input;
	i = 0;
	while (i < eng->n_draft && i < MTP_MAX_DRAFT)
	{
		logits = transformer_forward(eng->draft, curr_gemma, pos + i);
		max_id = 0;
		max_val = logits[0];
		for (j = 1; j < eng->draft->config.vocab_size; j++)
		{
			if (logits[j] > max_val)
			{
				max_val = logits[j];
				max_id = j;
			}
		}
		draft_tokens_gemma[i] = max_id;
		draft_tokens_target[i] = bridge_translate(&eng->bridge, max_id);
		curr_gemma = max_id;
		i++;
	}
	(void)draft_tokens_gemma;  /* Reserved for debug */
	eng->total_drafted += eng->n_draft;

	/* ====================================================================== */
	/* 3. VERIFICATION PHASE (Burst Output)                                   */
	/* ====================================================================== */
	n_out = 0;
	check_input = prompt_token;
	i = 0;
	while (i < eng->n_draft)
	{
		logits = transformer_forward(eng->target, check_input, pos + i);
		target_pred = 0;
		max_val = logits[0];
		for (j = 1; j < eng->target->config.vocab_size; j++)
		{
			if (logits[j] > max_val)
			{
				max_val = logits[j];
				target_pred = j;
			}
		}
		if (target_pred == draft_tokens_target[i])
		{
			/* ACCEPT: Add to output buffer */
			out_tokens[n_out++] = target_pred;
			eng->total_accepted++;
			check_input = target_pred;
		}
		else
		{
			/* REJECT: Add correct token and stop */
			out_tokens[n_out++] = target_pred;
			eng->total_rejected++;
			/* Rewind KV caches */
			kv_cache_rewind(eng->target, pos + n_out);
			kv_cache_rewind(eng->draft, pos + n_out);
			return (n_out);
		}
		i++;
	}

	/* ====================================================================== */
	/* 4. FULL ACCEPT: All drafts correct                                     */
	/* ====================================================================== */
	/* n_out == n_draft, all tokens accepted */
	return (n_out);
}

/*
** Print MTP statistics.
*/
void	mtp_stats(t_mtp_engine *eng)
{
	float	accept_rate;
	float	speedup;

	if (!eng)
		return ;
	if (eng->total_drafted > 0)
		accept_rate = (float)eng->total_accepted / (float)eng->total_drafted
			* 100.0f;
	else
		accept_rate = 0.0f;
	/* Theoretical speedup: 1 + (n_draft - 1) * accept_rate */
	/* Example: n=5, 60% accept = 1 + 4*0.6 = 3.4x */
	speedup = 1.0f + (float)(eng->n_draft - 1) * (accept_rate / 100.0f);
	printf("[MTP] Stats: %d drafted, %d accepted (%.1f%%), %d rejected\n",
		eng->total_drafted, eng->total_accepted, accept_rate, eng->total_rejected);
	printf("[MTP] Theoretical speedup: %.2fx\n", speedup);
	bridge_stats(&eng->bridge);
}

/*
** Free MTP engine resources.
*/
void	mtp_free(t_mtp_engine *eng)
{
	if (!eng)
		return ;
	bridge_free(&eng->bridge);
	memset(eng, 0, sizeof(t_mtp_engine));
}
