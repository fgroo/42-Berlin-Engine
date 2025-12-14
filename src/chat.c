/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   chat.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include "compute/sampler.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <locale.h>

#define MAX_INPUT_LEN 4096
// MAX_GEN_LEN now in config.h
#define MAX_TOKENS 8192

// Stop strings for generation (model often doesn't emit EOS)
// NOTE: Removed "\n\n" - too aggressive, cuts off mid-generation
static const char *g_stop_strings[] = {
	"</s>",
	"User:",
	"[/INST]",
	"[INST]",
	// "\n\n",  // REMOVED: causes premature stop after "Hello!\n\n"
	"\n\nUser",
	NULL
};

// GLOBAL SESSION STATE: Persistent position counter for RoPE continuity
static int g_session_pos = 0;

// Special token IDs (from tokenizer_config.json)
#define TOKEN_BOS 1           // <s>
#define TOKEN_EOS 2           // </s>
#define TOKEN_INST 3          // [INST]
#define TOKEN_INST_END 4      // [/INST]
#define TOKEN_SYS 17          // [SYSTEM_PROMPT]
#define TOKEN_SYS_END 18      // [/SYSTEM_PROMPT]
#define TOKEN_THINK 34        // [THINK]
#define TOKEN_THINK_END 35    // [/THINK]

// Minimal system prompt
#define SYSTEM_PROMPT ""

// Trim leading and trailing whitespace in-place
static void trim_whitespace(char *str)
{
	char	*start;
	char	*end;
	size_t	len;

	if (!str || !str[0])
		return;
	// Trim leading whitespace
	start = str;
	while (*start && (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r'))
		start++;
	// Trim trailing whitespace
	len = strlen(start);
	if (len == 0)
	{
		str[0] = '\0';
		return;
	}
	end = start + len - 1;
	while (end > start && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r'))
		end--;
	// Shift to start and null terminate
	len = end - start + 1;
	memmove(str, start, len);
	str[len] = '\0';
}

// ==================== UTF-8 SAFE PRINT BUFFER ====================
// LLM tokens can split UTF-8 multi-byte characters across boundaries.
// This buffer accumulates bytes until complete UTF-8 sequences are formed.
static unsigned char	g_utf8_buf[256];
static int				g_utf8_len = 0;

// Returns expected byte count for UTF-8 lead byte
// Returns 0 for continuation bytes (0x80-0xBF) - these can't start a sequence
static int	utf8_char_len(unsigned char c)
{
	if ((c & 0x80) == 0x00)
		return (1);  // ASCII
	if ((c & 0xC0) == 0x80)
		return (0);  // Continuation byte - invalid as lead
	if ((c & 0xE0) == 0xC0)
		return (2);
	if ((c & 0xF0) == 0xE0)
		return (3);
	if ((c & 0xF8) == 0xF0)
		return (4);
	return (1);  // Invalid, treat as single byte
}

// Print as much complete UTF-8 as possible, keep incomplete trailing bytes
static void	print_token_utf8(const char *piece)
{
	int				i;
	int				len;
	int				print_end;
	int				expected;
	int				j;
	unsigned char	c;

	if (!piece || !piece[0])
		return ;
	
	// Add piece to buffer
	len = strlen(piece);
	for (i = 0; i < len && g_utf8_len < 250; i++)
		g_utf8_buf[g_utf8_len++] = (unsigned char)piece[i];
	
	// Find how many bytes we can safely print (complete UTF-8 sequences)
	print_end = 0;
	i = 0;
	while (i < g_utf8_len)
	{
		c = g_utf8_buf[i];
		expected = utf8_char_len(c);
		
		if (expected == 0)
		{
			// Orphan continuation byte - skip it
			i++;
			print_end = i;
			continue;
		}
		
		if (i + expected <= g_utf8_len)
		{
			// Check that all continuation bytes are valid
			int valid = 1;
			for (j = 1; j < expected && valid; j++)
			{
				if ((g_utf8_buf[i + j] & 0xC0) != 0x80)
					valid = 0;
			}
			if (valid)
			{
				i += expected;
				print_end = i;
			}
			else
			{
				// Invalid sequence, skip the lead byte
				i++;
				print_end = i;
			}
		}
		else
		{
			// Incomplete sequence - stop here, keep for later
			break;
		}
	}
	
	// Print complete portion
	if (print_end > 0)
	{
		g_utf8_buf[print_end] = '\0';
		printf("%s", (char *)g_utf8_buf);
		fflush(stdout);
		
		// Shift remaining bytes to start
		if (print_end < g_utf8_len)
		{
			for (i = 0; i < g_utf8_len - print_end; i++)
				g_utf8_buf[i] = g_utf8_buf[print_end + i];
			g_utf8_len -= print_end;
		}
		else
		{
			g_utf8_len = 0;
		}
	}
}

// Flush any remaining bytes (call at end of generation)
static void	flush_utf8_buffer(void)
{
	if (g_utf8_len > 0)
	{
		g_utf8_buf[g_utf8_len] = '\0';
		printf("%s", (char *)g_utf8_buf);
		fflush(stdout);
		g_utf8_len = 0;
	}
}

// Build chat tokens with official Ministral-3B Reasoning format:
// First turn:  <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST]
// Later turns: [INST]{user}[/INST] (no BOS, no system prompt)
static int build_chat_tokens(t_tokenizer *tok, const char *user_input, int **out_tokens, int is_first_turn)
{
	int *user_tokens = NULL;
	int *sys_tokens = NULL;
	int n_user, n_sys = 0, total;
	int i, idx;
	
	// Official Ministral default system prompt (simplified for concise answers)
	static const char *sys_prompt = "Be concise. Answer directly with the result.";

	// Add leading space to user input to prevent token merging after [INST]
	// Without this, "[INST]Sally" becomes malformed tokens → "Ally" hallucination
	char *spaced_input = malloc(strlen(user_input) + 2);
	spaced_input[0] = ' ';
	strcpy(spaced_input + 1, user_input);
	
	// Encode user input
	n_user = tokenizer_encode(tok, spaced_input, &user_tokens);
	free(spaced_input);
	if (n_user < 0) return -1;
	
	// Encode system prompt on first turn
	if (is_first_turn) {
		n_sys = tokenizer_encode(tok, sys_prompt, &sys_tokens);
		if (n_sys < 0) n_sys = 0;
	}

	// First turn: BOS + SYS + sys_tokens + SYS_END + INST + user_tokens + INST_END
	// Later turns: INST + user_tokens + INST_END
	if (is_first_turn)
		total = 1 + 1 + n_sys + 1 + 1 + n_user + 1; // BOS + SYS + sys + SYS_END + INST + user + INST_END
	else
		total = 1 + n_user + 1;     // INST + user + INST_END
	
	*out_tokens = malloc(total * sizeof(int));
	
	idx = 0;
	if (is_first_turn) {
		(*out_tokens)[idx++] = TOKEN_BOS;      // <s>
		(*out_tokens)[idx++] = TOKEN_SYS;      // [SYSTEM_PROMPT]
		for (i = 0; i < n_sys; i++)
			(*out_tokens)[idx++] = sys_tokens[i];
		(*out_tokens)[idx++] = TOKEN_SYS_END;  // [/SYSTEM_PROMPT]
	}
	(*out_tokens)[idx++] = TOKEN_INST;         // [INST]
	for (i = 0; i < n_user; i++)
		(*out_tokens)[idx++] = user_tokens[i];
	(*out_tokens)[idx++] = TOKEN_INST_END;     // [/INST]
	
	if (sys_tokens) free(sys_tokens);
	free(user_tokens);
	return total;
}



// Helper to check for stop strings in accumulated output
static int check_stop_string(const char *output)
{
	int i;

	i = 0;
	while (g_stop_strings[i])
	{
		if (strstr(output, g_stop_strings[i]))
			return (1);
		i++;
	}
	return (0);
}

// Helper to run generation
// Returns 1 if output contains expected_answer (or if expected_answer is NULL), 0 otherwise
// Uses GLOBAL g_session_pos for RoPE continuity across turns
static int run_generation(t_transformer *t, t_tokenizer *tok, const char *input_text, const char *expected_answer, t_arena *sampler_arena)
{
	int *tokens = NULL;
	int is_first_turn = (g_session_pos == 0) ? 1 : 0;
	int n_tokens = build_chat_tokens(tok, input_text, &tokens, is_first_turn);
	if (n_tokens < 0) return 0;

	// DEBUG: Show position and token info
	printf("[DEBUG] Turn %s, pos=%d->%d, tokens=%d\n",
		is_first_turn ? "FIRST" : "CONT",
		g_session_pos, g_session_pos + n_tokens, n_tokens);
	// Show first 5 tokens AND the last token (should be Token 4 = [/INST])
	printf("[DEBUG] Tokens: [%d, %d, %d, %d, %d ... LAST=%d] (1=BOS, 3=INST, 4=/INST)\n",
		tokens[0], tokens[1], n_tokens > 2 ? tokens[2] : -1, 
		n_tokens > 3 ? tokens[3] : -1, n_tokens > 4 ? tokens[4] : -1,
		tokens[n_tokens - 1]);
	fflush(stdout);

	if (expected_answer) printf("Sanity Check: %s\n", input_text);
	else printf("\033[90m[Thinking] ");

	fflush(stdout);

	// Prefill: Start from current session position
	for (int i = 0; i < n_tokens - 1; i++)
	{
		transformer_forward(t, tokens[i], g_session_pos + i);
		
		// Sanity check for NaN on first token
		if (i == 0)
		{
			float max_val = -INFINITY;
			for (int v = 0; v < t->config.vocab_size; v++) {
				if (t->state.logits[v] > max_val) {
					max_val = t->state.logits[v];
				}
			}
			if (isnan(max_val)) {
				printf("FATAL: Logits contain NaN! Aborting.\n");
				exit(1);
			}
		}

		if (t->nested_learning && i > 0)
			transformer_backward_step(t, tokens[i], g_session_pos + i - 1);
	}

	int next_token = tokens[n_tokens - 1];
	// Use global position for RoPE continuity!
	int pos = g_session_pos + n_tokens - 1;
	int is_thinking = 1;
	int stop_hit = 0;
	
	char response_buffer[1024] = {0};
	int resp_len = 0;
	
	// Track generated tokens for repetition penalty
	int generated_tokens[MAX_GEN_LEN];
	int n_generated = 0;

	for (int i = 0; i < MAX_GEN_LEN; i++)
	{
		transformer_forward(t, next_token, pos);
		
		t_tensor logits_tensor;
		logits_tensor.data = t->state.logits;
		logits_tensor.size = t->config.vocab_size;
		logits_tensor.dtype = DTYPE_F32;
		
		// REPETITION PENALTY: Discourage repeating recent tokens
		// Penalize all tokens in prompt + previously generated
		{
			float *logits = t->state.logits;
			// Penalize input tokens
			for (int j = 0; j < n_tokens; j++)
			{
				if (logits[tokens[j]] > 0)
					logits[tokens[j]] /= REPETITION_PENALTY;
				else
					logits[tokens[j]] *= REPETITION_PENALTY;
			}
			// Penalize already-generated tokens (prevents "000..." loops)
			for (int j = 0; j < n_generated; j++)
			{
				if (logits[generated_tokens[j]] > 0)
					logits[generated_tokens[j]] /= REPETITION_PENALTY;
				else
					logits[generated_tokens[j]] *= REPETITION_PENALTY;
			}
		}
		
		// DEBUG: Show top-5 logits to diagnose model collapse
		if (i < 3) {
			float *logits = t->state.logits;
			int top5[5] = {0,0,0,0,0};
			float top5_val[5] = {-1e9,-1e9,-1e9,-1e9,-1e9};
			for (int j = 0; j < t->config.vocab_size; j++) {
				if (logits[j] > top5_val[4]) {
					top5_val[4] = logits[j]; top5[4] = j;
					// Bubble sort into place
					for (int k = 3; k >= 0; k--) {
						if (top5_val[k+1] > top5_val[k]) {
							float tv = top5_val[k]; int ti = top5[k];
							top5_val[k] = top5_val[k+1]; top5[k] = top5[k+1];
							top5_val[k+1] = tv; top5[k+1] = ti;
						}
					}
				}
			}
			printf("[LOGITS] Top5: %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f)\n",
				top5[0], top5_val[0], top5[1], top5_val[1], top5[2], top5_val[2],
				top5[3], top5_val[3], top5[4], top5_val[4]);
			fflush(stdout);
		}
		
		// BLOCK TOKEN 0 (UNK): Mask it to -inf before sampling
		// UNK corrupts the latent space and causes model collapse
		t->state.logits[0] = -1e9f;
		
		// GREEDY SAMPLING (DEBUG MODE) - use argmax instead of top_p
		next_token = sample_argmax(&logits_tensor);
		// Original: next_token = sample_top_p(&logits_tensor, TEMPERATURE, TOP_P, sampler_arena);
		arena_reset(sampler_arena);
		
		// Track generated token for repetition penalty
		if (n_generated < MAX_GEN_LEN)
			generated_tokens[n_generated++] = next_token;
		
		// PROMPT-ONLY LEARNING: Do NOT learn during generation
		// This prevents self-reinforcing garbage loops
		// Learning happens in prefill phase only (lines 101-102)

		if (next_token == TOKEN_EOS || stop_hit) break;
		
		if (next_token == TOKEN_THINK_END && is_thinking)
		{
			is_thinking = 0;
			flush_utf8_buffer();
			if (!expected_answer) printf("\033[0m\n[Answer] ");
			fflush(stdout);
			pos++;
			continue;
		}

		const char *piece = tokenizer_decode(tok, next_token);
		if (piece) {
			if (!is_thinking && expected_answer) {
				// Accumulate response for checking
				strncat(response_buffer, piece, sizeof(response_buffer) - resp_len - 1);
				resp_len += strlen(piece);
			}
			if (!expected_answer) {
				print_token_utf8(piece);
				// Accumulate for stop string detection
				if (resp_len < (int)sizeof(response_buffer) - 1) {
					strncat(response_buffer, piece, sizeof(response_buffer) - resp_len - 1);
					resp_len += strlen(piece);
				}
				// Check for stop strings
				if (check_stop_string(response_buffer))
					stop_hit = 1;
			}
		}
		
		pos++;
	}
	
	flush_utf8_buffer();
	if (!expected_answer) printf("\033[0m\n");
	
	// UPDATE GLOBAL SESSION POSITION for next turn!
	g_session_pos = pos + 1;
	
	free(tokens);
	
	// KEEP KV CACHE for multi-turn conversation context!
	// Do NOT reset: t->state.kv_cache[l].current_seq_len = 0;
	
	// Show persistent state message
	if (!expected_answer && t->nested_learning) {
		printf("[State] Fluid Weights UPDATED. Context preserved. Ready for next input.\n");
		
		// TRANSIENT LEARNING: Reset fluid weights after each turn
		// This prevents weight accumulation and mode collapse
		// The model "forgets" the specialization but keeps KV cache context
		for (int l = 0; l < t->config.n_layers && t->fluid_layers; l++) {
			if (t->fluid_layers[l].w2_weight && t->fluid_layers[l].w2_weight->data) {
				memset(t->fluid_layers[l].w2_weight->data, 0, 
					t->fluid_layers[l].w2_weight->size * sizeof(uint16_t));
			}
		}
		printf("[State] Fluid Weights RESET (transient mode). Fresh reasoning next turn.\n");
	}

	if (expected_answer) {
		printf("Output: %s\n", response_buffer);
		if (strstr(response_buffer, expected_answer)) return 1;
		return 0;
	}
	return 1;
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		fprintf(stderr, "Usage: %s <model_path> <config_path>\n", argv[0]);
		return (1);
	}

	// Enable UTF-8 output - try explicit UTF-8 locale
	if (!setlocale(LC_ALL, "en_US.UTF-8"))
	{
		if (!setlocale(LC_ALL, "C.UTF-8"))
		{
			setlocale(LC_ALL, "");
			fprintf(stderr, "Warning: UTF-8 locale not available. German chars may break.\n");
		}
	}

	const char *model_path = argv[1];
	const char *config_path = argv[2];

	// Initialize Transformer
	t_transformer t;
	printf("Initializing model...\n");
	if (transformer_init(&t, model_path, config_path) != 0)
	{
		fprintf(stderr, "Failed to initialize transformer\n");
		return (1);
	}
	
	
	// Nested learning now enabled in model.c

	// Initialize Tokenizer
	t_tokenizer tok;
	printf("Initializing tokenizer...\n");
	char tokenizer_path[1024];
	snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", config_path);
	char *p = strrchr(tokenizer_path, '/');
	if (p) strcpy(p + 1, "tokenizer.json");
	else strcpy(tokenizer_path, "tokenizer.json");

	if (tokenizer_init(&tok, tokenizer_path) != 0)
	{
		fprintf(stderr, "Failed to initialize tokenizer from %s\n", tokenizer_path);
		return (1);
	}

	// Initialize Sampler Scratch Arena
	t_arena sampler_arena;
	arena_init(&sampler_arena, 4 * 1024 * 1024);
	
	// DEBUG: Tokenizer check
	{
		int *debug_tokens = NULL;
		int n_debug = tokenizer_encode(&tok, "13 * 3", &debug_tokens);
		printf("Debug Token IDs for '13 * 3': ");
		for (int di = 0; di < n_debug; di++) printf("%d ", debug_tokens[di]);
		printf("\n");
		if (debug_tokens) free(debug_tokens);
	}

	// SANITY CHECK (disabled - go straight to chat)
	// printf("Running Sanity Check...\n");
	// if (!run_generation(&t, &tok, "What is 1 + 1?", "2", &sampler_arena))
	// {
	// 	fprintf(stderr, "Sanity Check Failed! Output did not contain '2'.\n");
	// 	return (1);
	// }
	// printf("Sanity Check Passed!\n");

	printf("Chat initialized. Type 'exit' to quit.\n");
	printf("Commands: 'learn' (enable), 'nolearn' (disable), 'reset' (new conversation)\n");

	char input[MAX_INPUT_LEN];
	
	while (1)
	{
		printf("\nUser: ");
		if (!fgets(input, sizeof(input), stdin)) break;
		
		input[strcspn(input, "\n")] = 0;
		trim_whitespace(input);  // Remove all leading/trailing whitespace
		if (input[0] == '\0') continue;  // Skip empty input
		if (strcmp(input, "exit") == 0) break;
		
		// Toggle learning commands
		if (strcmp(input, "learn") == 0) {
			t.nested_learning = 1;
			printf("[MODE] Learning ENABLED. Weights will be updated.\n");
			continue;
		}
		if (strcmp(input, "nolearn") == 0) {
			t.nested_learning = 0;
			printf("[MODE] Learning DISABLED. Weights frozen.\n");
			continue;
		}
		// Reset conversation (clear position and KV cache)
		if (strcmp(input, "reset") == 0) {
			g_session_pos = 0;
			for (int l = 0; l < t.config.n_layers; l++)
				t.state.kv_cache[l].current_seq_len = 0;
			printf("[MODE] Conversation RESET. KV cache cleared.\n");
			continue;
		}

		run_generation(&t, &tok, input, NULL, &sampler_arena);
	}

	// Clean up all resources - no memory leaks!
	tokenizer_free(&tok);
	arena_free(&sampler_arena);
	transformer_free(&t);
	return (0);
}
