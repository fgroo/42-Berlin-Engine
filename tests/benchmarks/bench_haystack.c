/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bench_haystack.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 19:50:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/14 19:50:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** NEEDLE IN A HAYSTACK BENCHMARK
** ==============================
** Tests sparse attention's ability to retrieve information from long context
** - Hides a passkey in Block 0
** - Fills with distractors (Blocks 1-N)
** - Queries for the passkey at the end
*/

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
#include "compute/ops_lsh.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <sys/time.h>

#define TOKEN_BOS 1
#define TOKEN_INST 3
#define TOKEN_INST_END 4
#define MAX_GEN 64

static double get_time_ms(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv)
{
	t_transformer	t;
	t_tokenizer		tok;
	char			tokenizer_path[1024];
	char			*p;
	int				*tokens;
	int				n_tokens;
	int				pos;
	int				next_token;
	int				i;
	float			*logits = NULL;
	t_tensor		logits_tensor;
	double			start_time;
	double			end_time;
	double			prefill_time;
	double			gen_time;
	int				gen_count;

	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model.safetensors> <config.json>\n", argv[0]);
		return 1;
	}
	setlocale(LC_ALL, "");
	
	/* Build haystack prompt */
	char prompt[100000];
	int offset = 0;
	
	/* The Needle (Block 0) */
	offset += sprintf(prompt + offset, "The secret passkey is '42-BERLIN-IS-AWESOME'. Remember this. ");
	
	/* The Haystack (100 distractor sentences = ~1200 tokens) */
	for (i = 0; i < 100; i++)
		offset += sprintf(prompt + offset, "The weather in Berlin is cloudy but the code is compiling. ");
	
	/* The Query */
	offset += sprintf(prompt + offset, "What is the secret passkey? Answer with ONLY the passkey.");
	
	printf("=== NEEDLE IN A HAYSTACK TEST ===\n");
	printf("Prompt length: %d chars\n", offset);

	/* Init model */
	if (transformer_init(&t, argv[1], argv[2]) != 0)
	{
		fprintf(stderr, "Failed to init model\n");
		return 1;
	}
	t.nested_learning = 0;  /* Disable learning for this test */
	t.use_paged_kv = 1;     /* PAGED MODE ON - Test sparse attention */
	
	/* Init tokenizer */
	strncpy(tokenizer_path, argv[1], sizeof(tokenizer_path) - 1);
	p = strrchr(tokenizer_path, '/');
	if (p)
		strcpy(p + 1, "tokenizer.json");
	else
		strcpy(tokenizer_path, "tokenizer.json");
	tokenizer_init(&tok, tokenizer_path);
	
	/* Tokenize */
	int *user_tokens;
	int n_user = tokenizer_encode(&tok, prompt, &user_tokens);
	printf("User tokens: %d\n", n_user);
	
	/* Build full prompt: BOS + INST + user + INST_END */
	n_tokens = 1 + 1 + n_user + 1;
	tokens = malloc(n_tokens * sizeof(int));
	tokens[0] = TOKEN_BOS;
	tokens[1] = TOKEN_INST;
	memcpy(&tokens[2], user_tokens, n_user * sizeof(int));
	tokens[n_tokens - 1] = TOKEN_INST_END;
	free(user_tokens);
	
	printf("Total input tokens: %d\n", n_tokens);
	printf("Expected blocks: ~%d\n", (n_tokens + 15) / 16);
	printf("\n");

	/* Prefill */
	printf("=== PREFILL ===\n");
	start_time = get_time_ms();
	for (i = 0; i < n_tokens; i++)
	{
		logits = transformer_forward(&t, tokens[i], i);
		if (i % 100 == 0)
			printf("  Token %d/%d (blocks: %d)\n", i, n_tokens, t.paged_kv[0].n_blocks);
	}
	end_time = get_time_ms();
	prefill_time = end_time - start_time;
	printf("Prefill complete: %.1f ms (%.1f T/s)\n", 
		prefill_time, n_tokens * 1000.0 / prefill_time);
	printf("Final block count: %d\n", t.paged_kv[0].n_blocks);
	printf("\n");
	
	/* Generation */
	printf("=== GENERATION ===\n");
	printf("[OUTPUT] ");
	fflush(stdout);
	start_time = get_time_ms();
	pos = n_tokens;
	gen_count = 0;
	logits_tensor.data = logits;
	logits_tensor.size = t.config.vocab_size;
	logits_tensor.dtype = DTYPE_F32;
	next_token = sample_argmax(&logits_tensor);
	
	while (gen_count < MAX_GEN && next_token != 2)  /* Not EOS */
	{
		const char *piece = tokenizer_decode(&tok, next_token);
		if (piece)
		{
			printf("%s", piece);
			fflush(stdout);
		}
		
		logits = transformer_forward(&t, next_token, pos);
		logits_tensor.data = logits;
		
		/* Apply repetition penalty */
		for (int rp = 0; rp < gen_count && rp < 64; rp++)
		{
			/* Track generated tokens (simple ring buffer) */
			/* For now, just penalize the previous token */
		}
		if (gen_count > 0)
		{
			int prev = next_token;
			if (logits[prev] > 0)
				logits[prev] /= 1.3f;  /* Strong penalty */
			else
				logits[prev] *= 1.3f;
		}
		
		next_token = sample_argmax(&logits_tensor);
		pos++;
		gen_count++;
	}
	end_time = get_time_ms();
	gen_time = end_time - start_time;
	printf("\n\n");
	
	printf("=== RESULTS ===\n");
	printf("Prefill: %.1f ms (%d tokens, %.2f T/s)\n", 
		prefill_time, n_tokens, n_tokens * 1000.0 / prefill_time);
	printf("Generation: %.1f ms (%d tokens, %.2f T/s)\n",
		gen_time, gen_count, gen_count * 1000.0 / gen_time);
	printf("Blocks used: %d\n", t.paged_kv[0].n_blocks);
	printf("Sparse threshold: %d blocks\n", SPARSE_BLOCKS_K);
	
	if (t.paged_kv[0].n_blocks > SPARSE_BLOCKS_K)
		printf("SPARSE PATH: Active (attending to %d blocks instead of %d)\n", 
			SPARSE_BLOCKS_K, t.paged_kv[0].n_blocks);
	else
		printf("DENSE PATH: Active (blocks <= threshold)\n");

	/* ====== LSH INTELLIGENCE REPORT (Phase 9: Atomic Stats) ====== */
	#if DEBUG_LSH
	{
		printf("\n=== LSH INTELLIGENCE REPORT (Thread-Safe) ===\n");
		printf("Total Sparse Queries: %lu\n", 
			(unsigned long)atomic_load(&t.lsh_stats.total_queries));
		
		uint64_t val_count = atomic_load(&t.lsh_stats.validation_count);
		uint64_t hits = atomic_load(&t.lsh_stats.topk_hits);
		uint64_t total = atomic_load(&t.lsh_stats.topk_total);
		if (val_count > 0 && total > 0)
		{
			float avg_recall = (float)hits / (float)total;
			printf("Validations:          %lu\n", (unsigned long)val_count);
			printf("Avg Recall:           %.1f%% (%s)\n", 
				avg_recall * 100.0f,
				avg_recall >= 0.80f ? "✅ OK" : "⚠️  LOW");
			if (avg_recall < 0.70f)
				printf("⚠️  CRITICAL: Hash functions may be miscorrelated!\n");
		}
		else
			printf("Avg Recall:           (no validations)\n");
		
		uint64_t k_samples = atomic_load(&t.lsh_stats.k_samples);
		uint64_t used_k = atomic_load(&t.lsh_stats.total_used_k);
		if (k_samples > 0)
		{
			float avg_k = (float)used_k / (float)k_samples;
			printf("Avg K Used:           %.1f / %d (saved %.1f%%)\n", 
				avg_k, t.sparse_k,
				(1.0f - avg_k / (float)t.sparse_k) * 100.0f);
		}
		printf("===============================\n");
	}
	#endif

	free(tokens);
	transformer_free(&t);
	return 0;
}
