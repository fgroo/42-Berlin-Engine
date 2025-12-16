/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   bench_perf.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 00:00:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/14 00:00:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** PERFORMANCE BENCHMARK - Tokens per Second Measurement
** ======================================================
** - No stdin, no user interaction
** - Greedy sampling (temperature=0) for determinism
** - Reports detailed timing: prefill + generation
** - Measures actual tokens/second throughput
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
#include <time.h>
#include <sys/time.h>

#define GEN_TOKENS 32
#define NUM_WARMUP 2
#define NUM_RUNS 5
#define TOKEN_BOS 1
#define TOKEN_EOS 2
#define TOKEN_INST 3
#define TOKEN_INST_END 4

/* Prompts of varying length - includes a long stress test */
static const char *g_prompts[] = {
	"What is 2 + 2?",
	"Explain the concept of recursion in programming.",
	/* STRESS TEST: Long prompt (~1500 tokens) for batched prefill benchmark */
	"You are a world-renowned expert in machine learning and artificial intelligence. "
	"I need you to provide a comprehensive analysis of the following topics. "
	"First, explain the fundamental architecture of transformer models, including "
	"the attention mechanism, positional encoding, and the role of layer normalization. "
	"Discuss the differences between encoder-only, decoder-only, and encoder-decoder "
	"architectures, providing examples of models in each category. "
	"Second, dive deep into the mathematical foundations of attention. "
	"Explain how queries, keys, and values are computed, the softmax operation, "
	"and the scaled dot-product attention formula. Discuss multi-head attention "
	"and its benefits for capturing different types of relationships. "
	"Third, analyze the challenges of training large language models. "
	"Cover topics such as gradient vanishing and exploding, the importance of "
	"learning rate scheduling, warmup periods, and various regularization techniques. "
	"Discuss mixed-precision training and its impact on memory and compute efficiency. "
	"Fourth, explore the concept of in-context learning and few-shot prompting. "
	"Explain how models can learn new tasks without explicit fine-tuning, "
	"the role of demonstrations in the prompt, and the limitations of this approach. "
	"Fifth, discuss the emerging field of test-time training and nested learning. "
	"Explain how models can adapt during inference, the use of fluid weights, "
	"and the challenges of maintaining stability during online learning. "
	"Sixth, analyze sparse attention mechanisms and their computational benefits. "
	"Cover sliding window attention, dilated attention, and LSH-based attention routing. "
	"Discuss the trade-offs between computational efficiency and attention quality. "
	"Seventh, explore the concept of mixture of experts (MoE) architectures. "
	"Explain how routing works, the benefits of conditional computation, "
	"and the challenges of load balancing across experts. "
	"Eighth, discuss the role of quantization in deploying large models. "
	"Cover INT8, INT4, and mixed-precision quantization techniques. "
	"Explain the impact on model quality and the methods for minimizing degradation. "
	"Ninth, analyze the memory bandwidth bottleneck in autoregressive generation. "
	"Explain why generation speed is limited by memory bandwidth rather than compute, "
	"and discuss techniques like speculative decoding and parallel sampling. "
	"Tenth, provide a roadmap for the future of AI model architectures. "
	"Discuss trends like state space models, mixture of depths, and neural memory. "
	"Explain how these approaches might address current limitations of transformers. "
	"Finally, summarize all key points in a clear, concise manner suitable for "
	"a technical audience familiar with machine learning concepts. "
	"Provide specific mathematical formulas where appropriate and cite key papers. "
	"This comprehensive analysis should demonstrate your deep understanding of "
	"modern AI architectures and their practical implications for deployment.",
	NULL
};

static double get_time_ms(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int build_prompt_tokens(t_tokenizer *tok, const char *prompt, int **out)
{
	int *user_tokens;
	int n_user;
	int total;
	int i, idx;

	n_user = tokenizer_encode(tok, prompt, &user_tokens);
	if (n_user < 0)
		return -1;
	total = 1 + 1 + n_user + 1;
	*out = malloc(total * sizeof(int));
	if (!*out) {
		free(user_tokens);
		return -1;
	}
	idx = 0;
	(*out)[idx++] = TOKEN_BOS;
	(*out)[idx++] = TOKEN_INST;
	for (i = 0; i < n_user; i++)
		(*out)[idx++] = user_tokens[i];
	(*out)[idx++] = TOKEN_INST_END;
	free(user_tokens);
	return total;
}

typedef struct {
	double prefill_ms;
	double gen_ms;
	int prefill_tokens;
	int gen_tokens;
} t_timing;

static t_timing run_timed(t_transformer *t, t_tokenizer *tok, const char *prompt)
{
	t_timing result = {0, 0, 0, 0};
	int *tokens;
	int n_tokens;
	int i, pos, next_token;
	t_tensor logits_tensor;
	double start, end;

	n_tokens = build_prompt_tokens(tok, prompt, &tokens);
	if (n_tokens < 0)
		return result;

	/* Prefill timing - Use batched path for efficiency */
	start = get_time_ms();
	
	/* 
	** BATCHED PREFILL: Use GEMM path for maximum throughput
	** Process in chunks of MAX_PREFILL_BATCH (64 tokens)
	** Each GEMM call loads weights ONCE, computes 64 tokens
	*/
	{
		int chunk_start = 0;
		int n_prefill = n_tokens - 1;
		while (chunk_start < n_prefill)
		{
			int chunk_size = n_prefill - chunk_start;
			if (chunk_size > MAX_PREFILL_BATCH)
				chunk_size = MAX_PREFILL_BATCH;
			forward_prefill_batch(t, tokens + chunk_start, chunk_size, chunk_start);
			chunk_start += chunk_size;
		}
	}
	
	end = get_time_ms();
	result.prefill_ms = end - start;
	result.prefill_tokens = n_tokens - 1;

	/* Generation timing (still per-token, memory-bound) */
	start = get_time_ms();
	next_token = tokens[n_tokens - 1];
	pos = n_tokens - 1;
	for (i = 0; i < GEN_TOKENS; i++) {
		transformer_forward(t, next_token, pos);
		logits_tensor.data = t->state.logits;
		logits_tensor.size = t->config.vocab_size;
		logits_tensor.dtype = DTYPE_F32;
		next_token = sample_argmax(&logits_tensor);
		if (next_token == TOKEN_EOS)
			break;
		pos++;
		result.gen_tokens++;
	}
	end = get_time_ms();
	result.gen_ms = end - start;

	/* Reset KV cache */
	for (i = 0; i < t->config.n_layers; i++)
		t->state.kv_cache[i].current_seq_len = 0;
	/* Also reset paged KV if enabled */
	if (t->use_paged_kv)
	{
		for (i = 0; i < t->config.n_layers; i++)
		{
			t->paged_kv[i].n_blocks = 0;
			t->paged_kv[i].n_tokens = 0;
		}
	}
	free(tokens);
	return result;
}

int main(int argc, char **argv)
{
	t_transformer t;
	t_tokenizer tok;
	char tokenizer_path[1024];
	char *p;
	int i, run;
	t_timing timing;
	double total_prefill_ms = 0, total_gen_ms = 0;
	int total_prefill_tokens = 0, total_gen_tokens = 0;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <model.safetensors> <config.json>\n", argv[0]);
		return 1;
	}
	setlocale(LC_ALL, "");

	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║         42-BERLIN-ENGINE PERFORMANCE BENCHMARK               ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");

	printf("Model: %s\n", argv[1]);
	printf("Config: %s\n", argv[2]);
	printf("Warmup runs: %d, Timed runs: %d\n", NUM_WARMUP, NUM_RUNS);
	printf("Tokens per generation: %d\n\n", GEN_TOKENS);
	fflush(stdout);

	if (transformer_init(&t, argv[1], argv[2]) != 0) {
		fprintf(stderr, "Failed to init transformer\n");
		return 1;
	}
	t.nested_learning = 0;
	
	printf("[CONFIG] dim=%d, n_layers=%d, n_heads=%d, head_dim=%d\n",
		t.config.dim, t.config.n_layers, t.config.n_heads, t.config.head_dim);
	printf("[CONFIG] vocab_size=%d, sparse_k=%d\n\n", t.config.vocab_size, t.sparse_k);

	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", argv[2]);
	p = strrchr(tokenizer_path, '/');
	if (p)
		strcpy(p + 1, "tokenizer.json");
	else
		strcpy(tokenizer_path, "tokenizer.json");
	if (tokenizer_init(&tok, tokenizer_path) != 0) {
		fprintf(stderr, "Failed to init tokenizer\n");
		return 1;
	}

	/* Warmup */
	printf("Warming up (%d runs)...\n", NUM_WARMUP);
	fflush(stdout);
	for (run = 0; run < NUM_WARMUP; run++) {
		run_timed(&t, &tok, g_prompts[0]);
		printf(".");
		fflush(stdout);
	}
	printf(" done.\n\n");

	/* Timed runs */
	printf("Running %d timed iterations...\n\n", NUM_RUNS);
	for (run = 0; run < NUM_RUNS; run++) {
		i = 0;
		while (g_prompts[i]) {
			timing = run_timed(&t, &tok, g_prompts[i]);
			total_prefill_ms += timing.prefill_ms;
			total_gen_ms += timing.gen_ms;
			total_prefill_tokens += timing.prefill_tokens;
			total_gen_tokens += timing.gen_tokens;
			i++;
		}
		printf("Run %d complete.\n", run + 1);
		fflush(stdout);
	}

	/* Results */
	printf("\n╔══════════════════════════════════════════════════════════════╗\n");
	printf("║                        RESULTS                               ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");

	double prefill_tps = (total_prefill_ms > 0) ? 
		(total_prefill_tokens / total_prefill_ms * 1000.0) : 0;
	double gen_tps = (total_gen_ms > 0) ?
		(total_gen_tokens / total_gen_ms * 1000.0) : 0;

	printf("Prefill:\n");
	printf("  Total tokens: %d\n", total_prefill_tokens);
	printf("  Total time:   %.2f ms\n", total_prefill_ms);
	printf("  Throughput:   %.2f tokens/sec\n\n", prefill_tps);

	printf("Generation:\n");
	printf("  Total tokens: %d\n", total_gen_tokens);
	printf("  Total time:   %.2f ms\n", total_gen_ms);
	printf("  Throughput:   %.2f tokens/sec\n\n", gen_tps);

	printf("═══════════════════════════════════════════════════════════════\n");
	printf("       GENERATION SPEED: %.2f T/s\n", gen_tps);
	printf("═══════════════════════════════════════════════════════════════\n");

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
			printf("Avg Recall:           %.1f%% (%s)\n", 
				avg_recall * 100.0f,
				avg_recall >= 0.80f ? "✅ OK" : "⚠️  LOW");
			printf("Validations:          %lu\n", (unsigned long)val_count);
		}
		else
			printf("Avg Recall:           (no validations)\n");
		
		uint64_t k_samples = atomic_load(&t.lsh_stats.k_samples);
		uint64_t used_k = atomic_load(&t.lsh_stats.total_used_k);
		if (k_samples > 0)
		{
			float avg_k = (float)used_k / (float)k_samples;
			printf("Avg K Used:           %.1f / %d\n", avg_k, t.sparse_k);
		}
		printf("===============================\n");
	}
	#endif

	tokenizer_free(&tok);
	transformer_free(&t);
	return 0;
}
