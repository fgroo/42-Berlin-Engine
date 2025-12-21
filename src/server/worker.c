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

	ctx = (t_worker_ctx *)arg;
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

		/* Phase 10: Chat Template Injection (Ministral-3B-Instruct format) */
		char	*final_prompt;
		char	template_buf[16384];
		int		template_allocated;

		template_allocated = 0;
		final_prompt = job.prompt;

		/* Check if user already sent the template */
		if (job.prompt && strstr(job.prompt, "[INST]") == NULL)
		{
			int required_len;

			/* Ministral format: [INST] {prompt} [/INST]
			** NOTE: No <s> prefix! Tokenizer adds BOS automatically.
			** Adding <s> here causes Double BOS which confuses the model. */
			required_len = snprintf(NULL, 0, "[INST] %s [/INST]", job.prompt);
			if (required_len < (int)sizeof(template_buf))
			{
				snprintf(template_buf, sizeof(template_buf),
					"[INST] %s [/INST]", job.prompt);
				final_prompt = template_buf;
			}
			else
			{
				/* Heap fallback for huge prompts */
				final_prompt = malloc(required_len + 1);
				if (final_prompt)
				{
					snprintf(final_prompt, required_len + 1,
						"[INST] %s [/INST]", job.prompt);
					template_allocated = 1;
				}
				else
					final_prompt = job.prompt;  /* Fallback to raw */
			}
			printf("[WORKER] Template injected\n");
		}
		printf("[WORKER] Final prompt: %.100s%s\n", final_prompt,
			strlen(final_prompt) > 100 ? "..." : "");

		/* 3. Tokenize prompt */
		tokens = NULL;
		n_tokens = tokenizer_encode(ctx->tokenizer, final_prompt, &tokens);
		if (n_tokens <= 0 || !tokens)
		{
			printf("[WORKER] Tokenization failed\n");
			close(job.client_fd);
			if (job.prompt)
				free(job.prompt);
			if (template_allocated)
				free(final_prompt);
			continue ;
		}
		printf("[WORKER] Tokenized %d tokens\n", n_tokens);

		/* 4. Prefill (batched) */
		if (n_tokens > 1)
			forward_prefill_batch(ctx->engine, tokens, n_tokens - 1, 0);

		/* 5. Get initial logits from last prompt token */
		logits = transformer_forward(ctx->engine, tokens[n_tokens - 1],
				n_tokens - 1);

		/* 6. Generation loop with streaming */
		full_len = 0;
		full_response[0] = '\0';
		gen_count = 0;
		max_gen = (job.max_tokens > 0) ? job.max_tokens : 256;
		if (max_gen > 1024)
			max_gen = 1024;

		while (gen_count < max_gen)
		{
			/* Argmax sampling (deterministic) */
			int		max_idx;
			float	max_val;
			int		j;

			max_idx = 0;
			max_val = logits[0];
			j = 1;
			while (j < ctx->engine->config.vocab_size)
			{
				if (logits[j] > max_val)
				{
					max_val = logits[j];
					max_idx = j;
				}
				j++;
			}
			next_token = max_idx;

			/* Check for EOS (token IDs vary by model) */
			if (next_token == 2 || next_token == 0)
				break ;

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

			/* Forward pass for next token */
			logits = transformer_forward(ctx->engine, next_token,
					n_tokens + gen_count);
			gen_count++;
		}

		/* 7. Finalize response */
		if (job.stream)
			send_sse_done(job.client_fd);
		else
			send_json_completion(job.client_fd, full_response, n_tokens,
				gen_count);

		printf("[WORKER] Generated %d tokens for socket %d\n",
			gen_count, job.client_fd);

		/* 8. Cleanup */
		close(job.client_fd);
		if (job.prompt)
			free(job.prompt);
		if (tokens)
			free(tokens);
		if (template_allocated)
			free(final_prompt);
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
