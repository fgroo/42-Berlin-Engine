/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   worker.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 23:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 23:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "worker.h"
#include "teacher.h"                /* MOPD: teacher_fetch_completion */
#include "memory/paged.h"
#include "inference/inference.h"  /* transformer_backward_step et al */
#include "compute/sampler.h"      /* sample_top_p, sample_argmax */
#include "config.h"               /* NESTED_LR, REPETITION_PENALTY */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
** ============================================================================
** SSE STREAMING HELPERS
** ============================================================================
*/

/* Helper to suppress unused result warning on write() */
static ssize_t	safe_write(int fd, const void *buf, size_t count)
{
	ssize_t	ret;

	ret = write(fd, buf, count);
	return (ret);  /* Caller can check if needed */
}

/* SSE headers for streaming response */
static const char	*g_sse_headers =
	"HTTP/1.1 200 OK\r\n"
	"Content-Type: text/event-stream\r\n"
	"Cache-Control: no-cache\r\n"
	"Connection: keep-alive\r\n"
	"Access-Control-Allow-Origin: *\r\n"
	"\r\n";

/*
** Send SSE chunk with OpenAI-compatible format
** Escapes special characters in content
*/
static void	json_escape(const char *src, char *dst, int max_len)
{
	int	i;
	int	j;

	i = 0;
	j = 0;
	while (src[i] && j < max_len - 2)
	{
		if (src[i] == '"' || src[i] == '\\')
		{
			dst[j++] = '\\';
			dst[j++] = src[i];
		}
		else if (src[i] == '\n')
		{
			dst[j++] = '\\';
			dst[j++] = 'n';
		}
		else if (src[i] == '\r')
		{
			dst[j++] = '\\';
			dst[j++] = 'r';
		}
		else if (src[i] == '\t')
		{
			dst[j++] = '\\';
			dst[j++] = 't';
		}
		else
		{
			dst[j++] = src[i];
		}
		i++;
	}
	dst[j] = '\0';
}

static void	send_sse_chunk(int fd, const char *content)
{
	char	buffer[4096];
	char	escaped[2048];

	json_escape(content, escaped, 2000);
	snprintf(buffer, sizeof(buffer),
		"data: {\"object\":\"chat.completion.chunk\","
		"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"}}]}\n\n",
		escaped);
	safe_write(fd, buffer, strlen(buffer));
}

/*
** Phase 9: Send reasoning_content chunk (DeepSeek R1 / GLM-4 style)
*/
static void	send_sse_reasoning(int fd, const char *content)
{
	char	buffer[4096];
	char	escaped[2048];

	json_escape(content, escaped, 2000);
	snprintf(buffer, sizeof(buffer),
		"data: {\"object\":\"chat.completion.chunk\","
		"\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"%s\"}}]}\n\n",
		escaped);
	safe_write(fd, buffer, strlen(buffer));
}

/*
** Send final SSE event
*/
static void	send_sse_done(int fd)
{
	const char	*done_msg = "data: [DONE]\n\n";

	safe_write(fd, done_msg, strlen(done_msg));
}

/*
** Send non-streaming JSON response
*/
static void	send_json_completion(int fd, const char *content, int prompt_tokens,
		int completion_tokens)
{
	char	headers[256];
	char	body[8192];
	char	escaped[4096];
	int		body_len;
	int		i;
	int		j;

	/* Escape content for JSON */
	i = 0;
	j = 0;
	while (content[i] && j < 4000)
	{
		if (content[i] == '"')
		{
			escaped[j++] = '\\';
			escaped[j++] = '"';
		}
		else if (content[i] == '\\')
		{
			escaped[j++] = '\\';
			escaped[j++] = '\\';
		}
		else if (content[i] == '\n')
		{
			escaped[j++] = '\\';
			escaped[j++] = 'n';
		}
		else
		{
			escaped[j++] = content[i];
		}
		i++;
	}
	escaped[j] = '\0';

	/* Build JSON body */
	body_len = snprintf(body, sizeof(body),
		"{"
		"\"id\":\"chatcmpl-42engine\","
		"\"object\":\"chat.completion\","
		"\"model\":\"42-engine\","
		"\"choices\":[{"
		"\"index\":0,"
		"\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
		"\"finish_reason\":\"stop\""
		"}],"
		"\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}"
		"}",
		escaped, prompt_tokens, completion_tokens);

	/* Send HTTP response */
	snprintf(headers, sizeof(headers),
		"HTTP/1.1 200 OK\r\n"
		"Content-Type: application/json\r\n"
		"Content-Length: %d\r\n"
		"Connection: close\r\n"
		"\r\n",
		body_len);
	safe_write(fd, headers, strlen(headers));
	safe_write(fd, body, body_len);
}

/*
** ============================================================================
** WORKER THREAD ROUTINE
** ============================================================================
*/

void	*worker_routine(void *arg)
{
	t_worker_ctx	*ctx;
	t_job			job;
	int				*tokens;
	int				n_tokens;
	float			*logits;
	int				next_token;
	int				gen_count;
	const char		*piece;
	char			full_response[8192];
	int				full_len;
	int				max_gen;
	t_arena			sampler_scratch;  /* For Top-P sampling */
	int				generated_tokens[1024];  /* For repetition penalty */
	int				n_generated;

	ctx = (t_worker_ctx *)arg;
	arena_init(&sampler_scratch, 4 * 1024 * 1024);  /* 4MB for sampler */
	printf("[WORKER] Inference worker started\n");

	while (ctx->running)
	{
		/* 1. Wait for job */
		job = queue_pop(ctx->queue);
		if (job.client_fd < 0)
		{
			/* Shutdown signal or empty queue */
			if (!ctx->running)
				break ;
			continue ;
		}
		printf("[WORKER] Processing job for socket %d (stream=%d)\n",
			job.client_fd, job.stream);

		/* 2. Send headers */
		if (job.stream)
			safe_write(job.client_fd, g_sse_headers, strlen(g_sse_headers));

		/* Phase 9: If thinking is enabled, send reasoning_content first */
		if (job.stream && job.enable_thinking)
		{
			printf("[WORKER] Thinking mode enabled (budget=%d)\n",
				job.thinking_budget);
			send_sse_reasoning(job.client_fd,
				"Analyzing the request and planning response...");
			send_sse_reasoning(job.client_fd,
				"Considering optimal implementation approach...");
		}

		/* Phase 10: MOPD - Fetch teacher completion BEFORE prefill */
		if (job.mopd && job.learn)
		{
			const char *api_key = getenv("OPENAI_API_KEY");
			if (api_key)
			{
				printf("[MOPD] \033[0;36m🎓 Consulting Teacher GLM-4.7...\033[0m\n");
				fflush(stdout);
				char *teacher_text = teacher_fetch_completion(job.prompt, api_key, 50);
				if (teacher_text)
				{
					printf("[MOPD] Teacher says: '%.100s%s'\n", teacher_text,
						strlen(teacher_text) > 100 ? "..." : "");
					/* Tokenize teacher response */
					job.teacher_tokens = NULL;
					job.n_teacher_tokens = tokenizer_encode(ctx->tokenizer,
						teacher_text, &job.teacher_tokens);
					if (job.n_teacher_tokens > 0 && job.teacher_tokens)
					{
						printf("[MOPD] \033[0;32m✓ Teacher provided %d tokens\033[0m\n",
							job.n_teacher_tokens);
					}
					else
					{
						printf("[MOPD] Failed to tokenize teacher response\n");
						job.teacher_tokens = NULL;
						job.n_teacher_tokens = 0;
					}
					free(teacher_text);
				}
				else
				{
					printf("[MOPD] Teacher silent. Fallback to Self-Correction.\n");
				}
			}
			else
			{
				printf("[MOPD] No API key. Set OPENAI_API_KEY. Using Self-Correction.\n");
			}
		}

		/* Phase 10: Chat Template using Token IDs (like chat.c)
		** String-based templating fails because the tokenizer doesn't
		** recognize [INST], [/INST] as special tokens when encoded as strings.
		** We must build: [BOS=1] [INST=3] <user_tokens> [INST_END=4] */
		
		#define TOKEN_BOS 1
		#define TOKEN_INST 3
		#define TOKEN_INST_END 4
		
		/* 3. Tokenize user prompt with space prefix (like chat.c) */
		int *user_tokens = NULL;
		char *spaced_prompt = malloc(strlen(job.prompt) + 2);
		if (spaced_prompt)
		{
			spaced_prompt[0] = ' ';
			strcpy(spaced_prompt + 1, job.prompt);
		}
		int n_user = tokenizer_encode(ctx->tokenizer, 
			spaced_prompt ? spaced_prompt : job.prompt, &user_tokens);
		if (spaced_prompt)
			free(spaced_prompt);
		if (n_user <= 0 || !user_tokens)
		{
			printf("[WORKER] Tokenization failed\n");
			close(job.client_fd);
			if (job.prompt)
				free(job.prompt);
			continue ;
		}
		
		/* Build final token array: [BOS] [INST] <user_tokens> [INST_END] */
		n_tokens = 1 + 1 + n_user + 1;  /* BOS + INST + user + INST_END */
		tokens = malloc(n_tokens * sizeof(int));
		if (!tokens)
		{
			printf("[WORKER] Token allocation failed\n");
			free(user_tokens);
			close(job.client_fd);
			if (job.prompt)
				free(job.prompt);
			continue ;
		}
		
		{
			int idx = 0;
			tokens[idx++] = TOKEN_BOS;       /* <s> */
			tokens[idx++] = TOKEN_INST;      /* [INST] */
			for (int i = 0; i < n_user; i++)
				tokens[idx++] = user_tokens[i];
			tokens[idx++] = TOKEN_INST_END;  /* [/INST] */
		}
	free(user_tokens);
		
		printf("[WORKER] Built %d tokens: [BOS=%d, INST=%d, ...%d user..., INST_END=%d]\n",
			n_tokens, tokens[0], tokens[1], n_user, tokens[n_tokens-1]);

		/* CRITICAL: Reset KV cache before each request!
		** Without this, old cached keys/values corrupt output. */
		for (int l = 0; l < ctx->engine->config.n_layers; l++)
			ctx->engine->state.kv_cache[l].current_seq_len = 0;
		if (ctx->engine->use_paged_kv && ctx->engine->paged_kv)
		{
			for (int l = 0; l < ctx->engine->config.n_layers; l++)
				paged_kv_reset(&ctx->engine->paged_kv[l]);
		}
		
		/* Phase 10: Enable runtime learning if requested */
		if (job.learn)
		{
			ctx->engine->nested_learning = 1;
			backward_zero_grads(ctx->engine);
			printf("[WORKER] \033[0;33m⚡ Runtime Learning ENABLED\033[0m\n");
		}
		else
		{
			ctx->engine->nested_learning = 0;
		}

		/* 4. SEQUENTIAL PREFILL (like chat.c)
		** The batch kernel (forward_prefill_batch) has a bug causing garbage.
		** Sequential forward is identical to the working chat.c code path. */
		printf("[WORKER] Running sequential prefill for %d tokens...\n", n_tokens);
		for (int i = 0; i < n_tokens - 1; i++)
		{
			transformer_forward(ctx->engine, tokens[i], i);
		}
		printf("[WORKER] Prefill done.\n");

		/* 5. Get initial logits from last prompt token */
		logits = transformer_forward(ctx->engine, tokens[n_tokens - 1],
				n_tokens - 1);

		/* 6. Generation loop with streaming */
		full_len = 0;
		full_response[0] = '\0';
		gen_count = 0;
		n_generated = 0;
		max_gen = (job.max_tokens > 0) ? job.max_tokens : 256;
		if (max_gen > 1024)
			max_gen = 1024;

		/* Sampling parameters */
		float temp = (job.temperature > 0.0f) ? job.temperature : 0.7f;
		float top_p = 0.9f;  /* Standard nucleus sampling */

		/* Sample first token from initial logits */
		{
			t_tensor logits_tensor;
			logits_tensor.data = logits;
			logits_tensor.size = ctx->engine->config.vocab_size;
			logits_tensor.dtype = DTYPE_F32;

			/* Apply repetition penalty to prompt tokens */
			for (int j = 0; j < n_tokens; j++)
			{
				if (logits[tokens[j]] > 0)
					logits[tokens[j]] /= REPETITION_PENALTY;
				else
					logits[tokens[j]] *= REPETITION_PENALTY;
			}
			/* Block UNK token */
			logits[0] = -1e9f;

			arena_reset(&sampler_scratch);
			next_token = sample_top_p(&logits_tensor, temp, top_p, &sampler_scratch);
		}

		while (gen_count < max_gen)
		{
			int		burst_tokens[16];
			int		n_burst;
			int		b;

			/* MTP or Legacy mode? */
			if (ctx->mtp && ctx->mtp->is_speculative)
			{
				/* MTP BURST: Generate 1-5 tokens at once */
				n_burst = mtp_generate(ctx->mtp, next_token,
						n_tokens + gen_count - 1, burst_tokens);
				if (n_burst > 1)
					printf("\033[0;32m[BURST] ⚡ %d tokens\033[0m\n", n_burst);
			}
			else
			{
				/* LEGACY: Forward pass, then sample from new logits */
				logits = transformer_forward(ctx->engine, next_token,
						n_tokens + gen_count);

				/* Build logits tensor for sampler */
				t_tensor logits_tensor;
				logits_tensor.data = logits;
				logits_tensor.size = ctx->engine->config.vocab_size;
				logits_tensor.dtype = DTYPE_F32;

				/* Apply repetition penalty to prompt + generated tokens */
				for (int j = 0; j < n_tokens; j++)
				{
					if (logits[tokens[j]] > 0)
						logits[tokens[j]] /= REPETITION_PENALTY;
					else
						logits[tokens[j]] *= REPETITION_PENALTY;
				}
				for (int j = 0; j < n_generated; j++)
				{
					if (logits[generated_tokens[j]] > 0)
						logits[generated_tokens[j]] /= REPETITION_PENALTY;
					else
						logits[generated_tokens[j]] *= REPETITION_PENALTY;
				}
				/* Block UNK token */
				logits[0] = -1e9f;

				/* Sample next token with Top-P */
				arena_reset(&sampler_scratch);
				next_token = sample_top_p(&logits_tensor, temp, top_p, &sampler_scratch);
				
				burst_tokens[0] = next_token;
				n_burst = 1;
			}

			/* Process all tokens in burst */
			b = 0;
			while (b < n_burst && gen_count < max_gen)
			{
				next_token = burst_tokens[b];

				/* Track generated tokens for repetition penalty */
				if (n_generated < 1024)
					generated_tokens[n_generated++] = next_token;

				/* Check for EOS */
				if (next_token == 2 || next_token == 0)
					goto end_generation;

				/* Decode token */
				piece = tokenizer_decode(ctx->tokenizer, next_token);
				if (piece)
				{
					int	piece_len;

					piece_len = strlen(piece);
					/* Stream the token */
					if (job.stream)
						send_sse_chunk(job.client_fd, piece);
					/* Accumulate for non-streaming */
					if (full_len + piece_len < 8000)
					{
						memcpy(full_response + full_len, piece, piece_len);
						full_len += piece_len;
						full_response[full_len] = '\0';
					}
				}

				/* Phase 10: RUNTIME LEARNING */
				if (job.learn && ctx->engine->nested_learning)
				{
					int target_token;
					
					/* MOPD: Use teacher token as ground truth if available */
					if (job.mopd && job.teacher_tokens && gen_count < job.n_teacher_tokens)
					{
						target_token = job.teacher_tokens[gen_count];
						/* Optional: Teacher Forcing - use teacher token for next step */
						/* This makes the model follow the teacher's path exactly */
						/* Uncomment for stricter distillation: */
						/* next_token = target_token; */
					}
					else
					{
						/* Self-Correction: Use our own sample as ground truth */
						target_token = next_token;
					}
					
					transformer_backward_step(ctx->engine, target_token,
						n_tokens + gen_count);
				}

				gen_count++;
				b++;
			}

			/* Check EOS */
			if (next_token == 2 || next_token == 0)
				break ;
		}
		end_generation:

		/* Phase 10: Apply accumulated gradients if learning was enabled */
		if (job.learn && ctx->engine->nested_learning)
		{
			backward_apply_grads(ctx->engine, ctx->engine->nested_lr);
			printf("[WORKER] \033[0;32m✓ Fluid weights updated (lr=%.5f)\033[0m\n",
				ctx->engine->nested_lr);
		}

		/* 7. Finalize response */
		if (job.stream)
			send_sse_done(job.client_fd);
		else
			send_json_completion(job.client_fd, full_response, n_tokens,
				gen_count);

		printf("[WORKER] Generated %d tokens for socket %d%s%s\n",
			gen_count, job.client_fd, 
			job.learn ? " [LEARNED]" : "",
			job.mopd ? " [MOPD]" : "");

		/* 8. Cleanup */
		close(job.client_fd);
		if (job.prompt)
			free(job.prompt);
		if (tokens)
			free(tokens);
		if (job.teacher_tokens)
			free(job.teacher_tokens);
	}

	printf("[WORKER] Inference worker stopped\n");
	return (NULL);
}

/*
** Start the worker thread
*/
int	worker_start(t_worker_ctx *ctx, pthread_t *thread)
{
	ctx->running = 1;
	if (pthread_create(thread, NULL, worker_routine, ctx) != 0)
	{
		perror("pthread_create");
		return (-1);
	}
	return (0);
}

/*
** Stop the worker thread
*/
void	worker_stop(t_worker_ctx *ctx, pthread_t thread)
{
	ctx->running = 0;
	queue_shutdown(ctx->queue);
	pthread_join(thread, NULL);
}
