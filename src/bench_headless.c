/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bench_headless.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/13 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/13 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** HEADLESS BENCHMARK - Non-Interactive Test Environment
** ======================================================
** - No stdin, no user interaction
** - Greedy sampling (temperature=0) for determinism
** - Static prompt list for regression testing
** - Max 20 tokens per prompt, stops at EOS
*/

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>

#define MAX_GEN_TOKENS 50
#define TOKEN_BOS 1
#define TOKEN_EOS 2
#define TOKEN_INST 3
#define TOKEN_INST_END 4

/* Test prompts - clearer instructions for math */
static const char *g_test_prompts[] = {
	"What is 1 plus 1? Answer with just the number.",
	"What is 2 times 3? Give the answer only.",
	"What is the capital of France?",
	"What color is the sky?",
	NULL
};

/* Build simple instruction tokens: <s>[INST] prompt [/INST] */
static int build_prompt_tokens(t_tokenizer *tok, const char *prompt,
								int **out_tokens)
{
	int		*user_tokens;
	int		n_user;
	int		total;
	int		i;
	int		idx;

	n_user = tokenizer_encode(tok, prompt, &user_tokens);
	if (n_user < 0)
		return (-1);
	total = 1 + 1 + n_user + 1;  /* BOS + INST + user + INST_END */
	*out_tokens = malloc(total * sizeof(int));
	if (!*out_tokens)
	{
		free(user_tokens);
		return (-1);
	}
	idx = 0;
	(*out_tokens)[idx++] = TOKEN_BOS;
	(*out_tokens)[idx++] = TOKEN_INST;
	i = 0;
	while (i < n_user)
	{
		(*out_tokens)[idx++] = user_tokens[i];
		i++;
	}
	(*out_tokens)[idx++] = TOKEN_INST_END;
	free(user_tokens);
	return (total);
}

/* Run single prompt with greedy sampling */
static void run_prompt(t_transformer *t, t_tokenizer *tok, const char *prompt)
{
	int			*tokens;
	int			n_tokens;
	int			i;
	int			pos;
	int			next_token;
	t_tensor	logits_tensor;
	const char	*piece;

	printf("\n[PROMPT] \"%s\"\n", prompt);
	fflush(stdout);
	n_tokens = build_prompt_tokens(tok, prompt, &tokens);
	if (n_tokens < 0)
	{
		printf("ERROR: tokenization failed\n");
		return ;
	}
	/* Debug: Show token IDs */
	printf("[TOKENS] n=%d: ", n_tokens);
	for (i = 0; i < n_tokens && i < 10; i++)
		printf("%d ", tokens[i]);
	if (n_tokens > 10)
		printf("...");
	printf("\n[OUTPUT] ");
	fflush(stdout);
	/* Prefill all tokens except last */
	i = 0;
	while (i < n_tokens - 1)
	{
		transformer_forward(t, tokens[i], i);
		i++;
	}
	/* Generation loop - greedy (argmax) */
	next_token = tokens[n_tokens - 1];
	pos = n_tokens - 1;
	i = 0;
	while (i < MAX_GEN_TOKENS)
	{
		transformer_forward(t, next_token, pos);
		logits_tensor.data = t->state.logits;
		logits_tensor.size = t->config.vocab_size;
		logits_tensor.dtype = DTYPE_F32;
		next_token = sample_argmax(&logits_tensor);
		
		/* Stop on EOS */
		if (next_token == TOKEN_EOS)
		{
			printf("<EOS>");
			break ;
		}
		/* Decode and print */
		piece = tokenizer_decode(tok, next_token);
		if (piece)
		{
			printf("%s", piece);
			fflush(stdout);
		}
		pos++;
		i++;
	}
	printf("\n");
	
	/* Reset KV cache for next prompt (fresh context) */
	for (i = 0; i < t->config.n_layers; i++)
	{
		t->state.kv_cache[i].current_seq_len = 0;
		if (t->use_paged_kv)
			paged_kv_reset(&t->paged_kv[i]);
	}
	free(tokens);
}

int	main(int argc, char **argv)
{
	t_transformer	t;
	t_tokenizer		tok;
	char			tokenizer_path[1024];
	char			*p;
	int				i;

	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model.safetensors> <config.json>\n", argv[0]);
		return (1);
	}
	setlocale(LC_ALL, "");
	printf("=== HEADLESS BENCHMARK ===\n");
	printf("Model: %s\n", argv[1]);
	printf("Config: %s\n", argv[2]);
	printf("Sampling: GREEDY (argmax, deterministic)\n");
	printf("Max tokens per prompt: %d\n", MAX_GEN_TOKENS);
	fflush(stdout);
	
	/* Init transformer */
	printf("\nInitializing model...\n");
	if (transformer_init(&t, argv[1], argv[2]) != 0)
	{
		fprintf(stderr, "Failed to init transformer\n");
		return (1);
	}
	/* Disable nested learning for clean test */
	t.nested_learning = 0;
	
	/* Debug: Print RoPE theta to confirm */
	printf("[CONFIG] rope_theta = %.1f\n", t.config.rope_theta);
	printf("[CONFIG] n_layers = %d, n_heads = %d, head_dim = %d\n",
		t.config.n_layers, t.config.n_heads, t.config.head_dim);
	fflush(stdout);
	
	/* Init tokenizer */
	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", argv[2]);
	p = strrchr(tokenizer_path, '/');
	if (p)
		strcpy(p + 1, "tokenizer.json");
	else
		strcpy(tokenizer_path, "tokenizer.json");
	if (tokenizer_init(&tok, tokenizer_path) != 0)
	{
		fprintf(stderr, "Failed to init tokenizer from %s\n", tokenizer_path);
		return (1);
	}
	printf("Tokenizer loaded.\n");
	fflush(stdout);
	
	/* Run test prompts */
	printf("\n=== RUNNING TESTS ===\n");
	i = 0;
	while (g_test_prompts[i])
	{
		run_prompt(&t, &tok, g_test_prompts[i]);
		i++;
	}
	printf("\n=== BENCHMARK COMPLETE ===\n");
	
	transformer_free(&t);
	tokenizer_free(&tok);
	return (0);
}
