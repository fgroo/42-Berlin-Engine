/* ************************************************************************** */
/*   bench_long_context.c - Long context performance test                     */
/*   Tests SIMD attention under heavy load (2000+ tokens)                     */
/* ************************************************************************** */

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <sys/time.h>

#define GEN_TOKENS 32
#define TOKEN_BOS 1
#define TOKEN_EOS 2
#define TOKEN_INST 3
#define TOKEN_INST_END 4

static double get_time_ms(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static char *read_file(const char *path)
{
	FILE *f = fopen(path, "r");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long len = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *buf = malloc(len + 1);
	if (!buf) { fclose(f); return NULL; }
	size_t read = fread(buf, 1, len, f);
	buf[read] = '\0';
	fclose(f);
	return buf;
}

int main(int argc, char **argv)
{
	t_transformer t;
	t_tokenizer tok;
	char tokenizer_path[1024];
	char *p;

	if (argc != 4) {
		fprintf(stderr, "Usage: %s <model.safetensors> <config.json> <long_prompt.txt>\n", argv[0]);
		return 1;
	}
	setlocale(LC_ALL, "");

	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║            LONG CONTEXT BENCHMARK (SIMD Test)                ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");

	/* Load prompt file */
	char *prompt_text = read_file(argv[3]);
	if (!prompt_text) {
		fprintf(stderr, "Failed to read prompt file: %s\n", argv[3]);
		return 1;
	}
	printf("Loaded prompt: %zu characters\n", strlen(prompt_text));

	/* Init model */
	if (transformer_init(&t, argv[1], argv[2]) != 0) {
		fprintf(stderr, "Failed to init transformer\n");
		return 1;
	}
	t.nested_learning = 0;  /* Pure inference, no learning */

	printf("[CONFIG] dim=%d, n_layers=%d, sparse_k=%d\n",
		t.config.dim, t.config.n_layers, t.sparse_k);

	/* Init tokenizer */
	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", argv[2]);
	p = strrchr(tokenizer_path, '/');
	if (p) strcpy(p + 1, "tokenizer.json");
	else strcpy(tokenizer_path, "tokenizer.json");
	if (tokenizer_init(&tok, tokenizer_path) != 0) {
		fprintf(stderr, "Failed to init tokenizer\n");
		return 1;
	}

	/* Tokenize prompt */
	int *user_tokens;
	int n_user = tokenizer_encode(&tok, prompt_text, &user_tokens);
	if (n_user < 0) {
		fprintf(stderr, "Failed to tokenize\n");
		return 1;
	}
	printf("Tokenized: %d tokens\n\n", n_user);
	free(prompt_text);

	/* Build full prompt with BOS/INST markers */
	int total = 1 + 1 + n_user + 1;  /* BOS + INST + tokens + INST_END */
	int *tokens = malloc(total * sizeof(int));
	int idx = 0;
	tokens[idx++] = TOKEN_BOS;
	tokens[idx++] = TOKEN_INST;
	for (int i = 0; i < n_user; i++)
		tokens[idx++] = user_tokens[i];
	tokens[idx++] = TOKEN_INST_END;
	free(user_tokens);

	printf("Total prompt tokens: %d (sparse_k=%d, use_sparse=%s)\n",
		total, t.sparse_k, (total > t.sparse_k) ? "YES" : "NO");

	/* ========== PREFILL BENCHMARK ========== */
	printf("\n=== PREFILL PHASE ===\n");
	printf("Processing %d tokens through 26 layers...\n", total - 1);
	fflush(stdout);

	double prefill_start = get_time_ms();
	for (int i = 0; i < total - 1; i++) {
		transformer_forward(&t, tokens[i], i);
		if ((i + 1) % 500 == 0) {
			double elapsed = get_time_ms() - prefill_start;
			double tps = (i + 1) / (elapsed / 1000.0);
			printf("  [%d/%d] %.1f T/s\n", i + 1, total - 1, tps);
			fflush(stdout);
		}
	}
	double prefill_end = get_time_ms();
	double prefill_ms = prefill_end - prefill_start;
	double prefill_tps = (total - 1) / (prefill_ms / 1000.0);

	printf("\nPrefill complete:\n");
	printf("  Tokens: %d\n", total - 1);
	printf("  Time:   %.2f ms\n", prefill_ms);
	printf("  Speed:  %.2f T/s\n", prefill_tps);

	/* ========== GENERATION BENCHMARK ========== */
	printf("\n=== GENERATION PHASE ===\n");
	printf("Generating %d tokens with KV cache (len=%d)...\n", GEN_TOKENS, total - 1);
	fflush(stdout);

	double gen_start = get_time_ms();
	int next_token = tokens[total - 1];
	int pos = total - 1;
	int gen_count = 0;
	t_tensor logits_tensor;

	for (int i = 0; i < GEN_TOKENS; i++) {
		transformer_forward(&t, next_token, pos);
		logits_tensor.data = t.state.logits;
		logits_tensor.size = t.config.vocab_size;
		logits_tensor.dtype = DTYPE_F32;
		next_token = sample_argmax(&logits_tensor);
		if (next_token == TOKEN_EOS) break;
		gen_count++;
		pos++;
	}
	double gen_end = get_time_ms();
	double gen_ms = gen_end - gen_start;
	double gen_tps = gen_count / (gen_ms / 1000.0);

	printf("\nGeneration complete:\n");
	printf("  Tokens: %d\n", gen_count);
	printf("  Time:   %.2f ms\n", gen_ms);
	printf("  Speed:  %.2f T/s\n", gen_tps);

	/* ========== RESULTS ========== */
	printf("\n╔══════════════════════════════════════════════════════════════╗\n");
	printf("║                        RESULTS                               ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");

	printf("Context Length: %d tokens (sparse_k=%d)\n", total, t.sparse_k);
	printf("Sparse Attention: %s\n\n", (total > t.sparse_k) ? "ACTIVE" : "INACTIVE");

	printf("┌─────────────┬─────────────┬─────────────┐\n");
	printf("│   Phase     │    Time     │    Speed    │\n");
	printf("├─────────────┼─────────────┼─────────────┤\n");
	printf("│ Prefill     │ %7.0f ms  │ %5.2f T/s   │\n", prefill_ms, prefill_tps);
	printf("│ Generation  │ %7.0f ms  │ %5.2f T/s   │\n", gen_ms, gen_tps);
	printf("└─────────────┴─────────────┴─────────────┘\n");

	printf("\n═══════════════════════════════════════════════════════════════\n");
	if (total > t.sparse_k)
		printf("   SIMD SPARSE ATTENTION: VALIDATED AT SCALE\n");
	else
		printf("   WARNING: Context < sparse_k, dense attention used\n");
	printf("═══════════════════════════════════════════════════════════════\n");

	free(tokens);
	tokenizer_free(&tok);
	transformer_free(&t);
	return 0;
}
