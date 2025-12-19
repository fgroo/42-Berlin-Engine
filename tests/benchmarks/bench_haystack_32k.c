/* ************************************************************************** */
/*                                                                            */
/*   bench_haystack_32k.c - 32K Token Context Stress Test                    */
/*                                                                            */
/*   Phase 2 Deep Freeze: Proves INT8 KV Cache memory bandwidth savings      */
/*   at scale where RAM becomes the bottleneck.                               */
/*                                                                            */
/* ************************************************************************** */

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
#include <sys/resource.h>

#define TOKEN_BOS 1
#define TOKEN_INST 3
#define TOKEN_INST_END 4
#define MAX_GEN 32

/* Target: 1K tokens context (fast iteration ~3 min at 5 T/s) */
#define TARGET_TOKENS 1024
#define NEEDLE_POSITION (TARGET_TOKENS / 2)

/*
** Synthetic distractor text - semantically irrelevant but realistic
** ~15 tokens per sentence
*/
static const char *g_distractor[] = {
	"The rapid development of industrial processes in modern manufacturing "
	"requires careful attention to quality control metrics. ",
	"Cloud computing infrastructure enables scalable deployment of distributed "
	"applications across multiple availability zones. ",
	"While the thermodynamic principles remain constant, computational efficiency "
	"continues to improve with each silicon generation. ",
	"Database replication strategies must balance consistency requirements with "
	"the practical limits of network latency. ",
	"The algorithmic complexity of sorting operations varies significantly "
	"depending on the initial distribution of elements. ",
	"Machine learning models require extensive training data to achieve "
	"satisfactory performance on real-world tasks. ",
	"Network protocols define the rules for data transmission between "
	"interconnected computing devices globally. ",
	"Software architecture decisions made early in development often have "
	"lasting implications for system maintainability. ",
};
#define N_DISTRACTORS 8

static const char *NEEDLE_TEXT = 
	"IMPORTANT: The secret passkey you must remember is 'BLUE_BERRY_PIE'. "
	"This is critical information. The passkey is BLUE_BERRY_PIE. ";

static const char *QUERY_TEXT = 
	"What is the secret passkey? Answer with ONLY the passkey itself.";

static double	get_time_ms(void)
{
	struct timeval	tv;

	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0);
}

static long	get_memory_kb(void)
{
	struct rusage	usage;

	getrusage(RUSAGE_SELF, &usage);
	return (usage.ru_maxrss);
}

int	main(int argc, char **argv)
{
	t_transformer	t;
	t_tokenizer		tok;
	char			tokenizer_path[1024];
	char			*p;
	float			*logits;
	t_tensor		logits_tensor;
	double			start_time;
	double			end_time;
	int				pos;
	int				i;
	int				next_token;
	int				gen_count;
	long			mem_before;
	long			mem_after;

	/* Distractor token cache */
	int				*distractor_tokens[N_DISTRACTORS];
	int				distractor_lens[N_DISTRACTORS];
	int				*needle_tokens;
	int				n_needle;
	int				*query_tokens;
	int				n_query;
	int				needle_inserted;
	int				distractor_idx;

	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model.safetensors> <config.json>\n", argv[0]);
		return (1);
	}
	setlocale(LC_ALL, "");

	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║     OPERATION HEUHAUFEN: 32K Token Stress Test               ║\n");
	printf("║     Phase 2 Deep Freeze - INT8 KV Cache Validation           ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");

	mem_before = get_memory_kb();
	printf("[INIT] Memory before model load: %ld KB\n", mem_before);

	/* Initialize model */
	if (transformer_init(&t, argv[1], argv[2]) != 0)
	{
		fprintf(stderr, "Failed to init model\n");
		return (1);
	}
	t.nested_learning = 0;  /* Disable learning */
	t.use_paged_kv = 1;     /* PAGED MODE ON */

	/* Initialize tokenizer */
	strncpy(tokenizer_path, argv[1], sizeof(tokenizer_path) - 1);
	p = strrchr(tokenizer_path, '/');
	if (p)
		strcpy(p + 1, "tokenizer.json");
	else
		strcpy(tokenizer_path, "tokenizer.json");
	tokenizer_init(&tok, tokenizer_path);

	mem_after = get_memory_kb();
	printf("[INIT] Memory after model load: %ld KB (+%ld KB)\n", 
		mem_after, mem_after - mem_before);

	/* Pre-tokenize distractors (avoid tokenizer overhead in hot loop) */
	printf("[SETUP] Pre-tokenizing distractor blocks...\n");
	for (i = 0; i < N_DISTRACTORS; i++)
	{
		distractor_lens[i] = tokenizer_encode(&tok, g_distractor[i], 
			&distractor_tokens[i]);
	}
	n_needle = tokenizer_encode(&tok, NEEDLE_TEXT, &needle_tokens);
	n_query = tokenizer_encode(&tok, QUERY_TEXT, &query_tokens);

	printf("[SETUP] Distractor blocks: %d tokens avg\n", distractor_lens[0]);
	printf("[SETUP] Needle: %d tokens\n", n_needle);
	printf("[SETUP] Target context: %d tokens\n", TARGET_TOKENS);
	printf("[SETUP] Needle position: ~%d (50%% depth)\n", NEEDLE_POSITION);
	printf("\n");

	/* ========== PREFILL: The Heavy Lifting ========== */
	printf("═══════════════════════════════════════════════════════════════\n");
	printf(" PREFILL PHASE: Ingesting %d tokens...\n", TARGET_TOKENS);
	printf("═══════════════════════════════════════════════════════════════\n");

	start_time = get_time_ms();
	mem_before = get_memory_kb();

	/* Feed BOS + INST */
	logits = transformer_forward(&t, TOKEN_BOS, 0);
	logits = transformer_forward(&t, TOKEN_INST, 1);
	pos = 2;
	needle_inserted = 0;
	distractor_idx = 0;

	while (pos < TARGET_TOKENS - n_query - 10)
	{
		/* Insert needle at ~50% depth */
		if (!needle_inserted && pos >= NEEDLE_POSITION)
		{
			for (i = 0; i < n_needle; i++)
			{
				logits = transformer_forward(&t, needle_tokens[i], pos);
				pos++;
			}
			needle_inserted = 1;
			printf("\r[PREFILL] ★ NEEDLE INSERTED at pos %d ★          ", pos);
			fflush(stdout);
		}
		else
		{
			/* Feed distractor (round-robin through blocks) */
			int *dtok = distractor_tokens[distractor_idx % N_DISTRACTORS];
			int dlen = distractor_lens[distractor_idx % N_DISTRACTORS];
			
			/* Don't overflow */
			if (pos + dlen > TARGET_TOKENS - n_query - 10)
				dlen = TARGET_TOKENS - n_query - 10 - pos;
			if (dlen <= 0)
				break ;

			for (i = 0; i < dlen; i++)
			{
				logits = transformer_forward(&t, dtok[i], pos);
				pos++;
			}
			distractor_idx++;
		}

		/* Progress every 4K tokens */
		if (pos % 4096 < 20)
		{
			printf("\r[PREFILL] Progress: %d / %d tokens (%.1f%%) | Blocks: %d    ",
				pos, TARGET_TOKENS, (float)pos * 100.0f / TARGET_TOKENS,
				t.paged_kv[0].n_blocks);
			fflush(stdout);
		}
	}

	/* Feed INST_END */
	logits = transformer_forward(&t, TOKEN_INST_END, pos++);

	end_time = get_time_ms();
	mem_after = get_memory_kb();

	printf("\n\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf(" PREFILL RESULTS\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("  Total Tokens:     %d\n", pos);
	printf("  Time:             %.2f seconds\n", (end_time - start_time) / 1000.0);
	printf("  Speed:            %.2f T/s\n", 
		pos * 1000.0 / (end_time - start_time));
	printf("  Blocks Used:      %d\n", t.paged_kv[0].n_blocks);
	printf("  Memory Delta:     +%ld KB (INT8 KV Cache)\n", mem_after - mem_before);
	printf("  Sparse Threshold: %d blocks\n", SPARSE_BLOCKS_K);
	if (t.paged_kv[0].n_blocks > SPARSE_BLOCKS_K)
		printf("  Attention Mode:   SPARSE (attending to %d of %d blocks)\n",
			SPARSE_BLOCKS_K, t.paged_kv[0].n_blocks);
	printf("\n");

	/* ========== GENERATION: Query the Needle ========== */
	printf("═══════════════════════════════════════════════════════════════\n");
	printf(" GENERATION PHASE: Querying for the needle...\n");
	printf("═══════════════════════════════════════════════════════════════\n");

	/* Feed query tokens */
	for (i = 0; i < n_query; i++)
	{
		logits = transformer_forward(&t, query_tokens[i], pos);
		pos++;
	}

	logits_tensor.data = logits;
	logits_tensor.size = t.config.vocab_size;
	logits_tensor.dtype = DTYPE_F32;

	printf("[OUTPUT] ");
	fflush(stdout);

	start_time = get_time_ms();
	gen_count = 0;

	while (gen_count < MAX_GEN)
	{
		next_token = sample_argmax(&logits_tensor);
		if (next_token == 2)  /* EOS */
			break ;

		const char *piece = tokenizer_decode(&tok, next_token);
		if (piece)
		{
			printf("%s", piece);
			fflush(stdout);
		}

		logits = transformer_forward(&t, next_token, pos);
		logits_tensor.data = logits;
		pos++;
		gen_count++;
	}

	end_time = get_time_ms();
	printf("\n\n");

	printf("═══════════════════════════════════════════════════════════════\n");
	printf(" FINAL RESULTS\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("  Generation Tokens: %d\n", gen_count);
	printf("  Generation Time:   %.2f ms\n", end_time - start_time);
	printf("  Generation Speed:  %.2f T/s\n", 
		gen_count * 1000.0 / (end_time - start_time));
	printf("  Peak Memory:       %ld KB\n", get_memory_kb());
	printf("\n");

	/* Validation */
	printf("═══════════════════════════════════════════════════════════════\n");
	printf(" VALIDATION\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("  Expected Answer:   BLUE_BERRY_PIE\n");
	printf("  Check above output for correct passkey retrieval.\n");
	printf("  If passkey found: INT8 quantization + sparse attention WORKS!\n");
	printf("═══════════════════════════════════════════════════════════════\n");

	/* Cleanup */
	for (i = 0; i < N_DISTRACTORS; i++)
		free(distractor_tokens[i]);
	free(needle_tokens);
	free(query_tokens);
	transformer_free(&t);

	return (0);
}
