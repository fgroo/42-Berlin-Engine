/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   server.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 21:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 21:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "server.h"
#include "queue.h"
#include "worker.h"
#include "json_parse.h"
#include "memory/safe_alloc.h"
#include "core/types.h"  /* MOPD: t_sparse_prob */
#include "nested/persistence.h"  /* Phase 6: fluid_save/load */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>  /* expf for logprob -> prob conversion */
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#ifdef __linux__
# include <sys/epoll.h>
# define USE_EPOLL 1
#else
# include <sys/select.h>
# define USE_EPOLL 0
#endif

/*
** ============================================================================
** SOCKET HELPERS
** ============================================================================
*/

static int	set_nonblocking(int fd)
{
	int	flags;

	flags = fcntl(fd, F_GETFL, 0);
	if (flags == -1)
		return (-1);
	return (fcntl(fd, F_SETFL, flags | O_NONBLOCK));
}

static int	set_reuseaddr(int fd)
{
	int	opt;

	opt = 1;
	return (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)));
}

static int	set_tcp_nodelay(int fd)
{
	int	opt;

	opt = 1;
	return (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)));
}

/*
** ============================================================================
** CONNECTION MANAGEMENT
** ============================================================================
*/

static void	conn_init(t_client_conn *conn, int fd)
{
	memset(conn, 0, sizeof(*conn));
	conn->fd = fd;
	conn->state = CONN_STATE_READING;
}

static void	conn_reset(t_client_conn *conn)
{
	if (conn->write_buf)
	{
		free(conn->write_buf);
		conn->write_buf = NULL;
	}
	if (conn->request.body)
	{
		free(conn->request.body);
		conn->request.body = NULL;
	}
	if (conn->tokens)
	{
		free(conn->tokens);
		conn->tokens = NULL;
	}
	if (conn->fd >= 0)
		close(conn->fd);
	conn->fd = -1;
}

static t_client_conn	*find_free_conn(t_server *srv)
{
	int	i;

	i = 0;
	while (i < srv->max_conns)
	{
		if (srv->conns[i].fd < 0)
			return (&srv->conns[i]);
		i++;
	}
	return (NULL);
}

/*
** ============================================================================
** HTTP PARSING (Minimal - just what we need)
** ============================================================================
*/

int	parse_http_request(t_client_conn *conn)
{
	char	*header_end;
	char	*line;
	char	*body_start;
	int		header_len;

	/* Find end of headers */
	header_end = strstr(conn->read_buf, "\r\n\r\n");
	if (!header_end)
		return (0);  /* Need more data */
	header_len = (int)(header_end - conn->read_buf + 4);
	body_start = header_end + 4;

	/* Parse request line: METHOD PATH HTTP/1.x */
	line = conn->read_buf;
	if (sscanf(line, "%15s %255s", conn->request.method,
			conn->request.path) != 2)
		return (-1);  /* Malformed */

	/* [SECURITY PATCH #1] Content-Length Validation
	** Old code: atoi(cl + 15) - VULNERABLE!
	** - Negative values: atoi("-1") = -1 → bypasses checks
	** - Huge values: atoi("99999999999") → undefined behavior
	** - No digits: atoi("abc") = 0 → silent failure
	** New code: strtol with full validation. Trust no client. */
	conn->request.content_length = 0;
	{
		char	*cl;
		char	*endptr;
		long	val;
		
		cl = strstr(conn->read_buf, "Content-Length:");
		if (!cl)
			cl = strstr(conn->read_buf, "content-length:");
		if (cl)
		{
			val = strtol(cl + 15, &endptr, 10);
			/* No digits parsed? Malformed header. */
			if (endptr == cl + 15)
			{
				fprintf(stderr, "[SEC] Malformed Content-Length header\n");
				return (-1);
			}
			/* Negative? Hacker. */
			if (val < 0)
			{
				fprintf(stderr, "[SEC] Negative Content-Length: %ld\n", val);
				return (-1);
			}
			/* Too large? DoS attempt. */
			if (val > MAX_BODY_SIZE)
			{
				fprintf(stderr, "[SEC] Request too large: %ld bytes (max %d)\n", 
					val, MAX_BODY_SIZE);
				return (-1);
			}
			conn->request.content_length = (int)val;
		}
	}

	/* Parse Connection: header */
	conn->request.keep_alive = 1;  /* Default for HTTP/1.1 */
	if (strstr(conn->read_buf, "Connection: close"))
		conn->request.keep_alive = 0;

	/* Check if body is complete */
	if (conn->request.content_length > 0)
	{
		int	body_received;

		body_received = conn->read_len - header_len;
		if (body_received < conn->request.content_length)
			return (0);  /* Need more body data */
		/* Allocate and copy body */
		conn->request.body = xmalloc(conn->request.content_length + 1);
		memcpy(conn->request.body, body_start, conn->request.content_length);
		conn->request.body[conn->request.content_length] = '\0';
		conn->request.body_len = conn->request.content_length;
	}
	return (1);  /* Request complete */
}

/*
** ============================================================================
** RESPONSE HELPERS
** ============================================================================
*/

static const char	*http_status_text(int status)
{
	if (status == 200)
		return ("OK");
	if (status == 400)
		return ("Bad Request");
	if (status == 404)
		return ("Not Found");
	if (status == 500)
		return ("Internal Server Error");
	return ("Unknown");
}

void	send_json_response(t_client_conn *conn, int status, const char *json)
{
	size_t	json_len;
	size_t	buf_size;
	int		header_len;

	json_len = strlen(json);
	/* [SECURITY PATCH #2] Response Buffer Hardening
	** Old code: sprintf(conn->write_buf, ...) - VULNERABLE!
	** sprintf writes until done, ignoring buffer size → stack smash.
	** New code: snprintf with explicit size and truncation check. */
	buf_size = json_len + 512;
	conn->write_buf = xmalloc(buf_size);
	header_len = snprintf(conn->write_buf, buf_size,
		"HTTP/1.1 %d %s\r\n"
		"Content-Type: application/json\r\n"
		"Content-Length: %zu\r\n"
		"Connection: %s\r\n"
		"\r\n",
		status, http_status_text(status),
		json_len,
		conn->request.keep_alive ? "keep-alive" : "close");
	/* Defense-in-depth: detect truncation (should never happen with +512) */
	if (header_len < 0 || (size_t)header_len >= buf_size)
	{
		fprintf(stderr, "[SEC] Response header truncated! This should not happen.\n");
		conn->write_len = 0;
		conn->state = CONN_STATE_CLOSING;
		return ;
	}
	/* Safe to append JSON body */
	memcpy(conn->write_buf + header_len, json, json_len);
	conn->write_len = header_len + (int)json_len;
	conn->write_pos = 0;
	conn->state = CONN_STATE_WRITING;
}

void	send_error_response(t_client_conn *conn, int status, const char *msg)
{
	char	json[512];

	snprintf(json, sizeof(json),
		"{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\","
		"\"code\":\"%d\"}}", msg, status);
	send_json_response(conn, status, json);
}

void	send_sse_event(t_client_conn *conn, const char *event, const char *data)
{
	char	buf[4096];
	int		len;
	ssize_t	sent;

	(void)event;  /* event type unused for now, always "data" */
	len = snprintf(buf, sizeof(buf), "data: %s\n\n", data);
	sent = write(conn->fd, buf, len);
	(void)sent;  /* Ignore partial writes for SSE */
}



/*
** ============================================================================
** CHAT COMPLETION REQUEST HANDLER
** ============================================================================
*/
/*
** Phase 9: Robust JSON parsing with OpenAI-compatible request handling
** Uses tolerant parser that ignores unknown fields
*/
static int	handle_chat_completions(t_server *srv, t_client_conn *conn)
{
	t_chat_request	req;
	t_job			job;

	if (!conn->request.body)
	{
		send_error_response(conn, 400, "Missing request body");
		return (-1);
	}

	/* Parse with robust parser (ignores unknown fields like 'thinking') */
	if (chat_request_parse(conn->request.body, &req) != 0)
	{
		send_error_response(conn, 400, "Missing 'content' in messages");
		chat_request_free(&req);
		return (-1);
	}

	/* Create job for worker thread */
	memset(&job, 0, sizeof(job));
	job.client_fd = conn->fd;
	job.prompt = req.content;     /* Worker will free this */
	req.content = NULL;           /* Prevent double-free */
	job.stream = req.stream;
	job.max_tokens = req.max_tokens;
	job.temperature = req.temperature;
	job.enable_thinking = req.enable_thinking;
	job.thinking_budget = req.thinking_budget;
	job.learn = req.learn;        /* Phase 10: Runtime Learning */
	job.mopd = req.mopd;          /* Phase 10: MOPD mode */
	job.teacher_tokens = NULL;    /* Will be set by worker if mopd=1 or forced_target */
	job.n_teacher_tokens = 0;
	job.forced_target = req.force_response;  /* JSONL Teacher: Dataset is Boss */
	req.force_response = NULL;    /* Transfer ownership to job */

	/* Log with thinking and learning status */
	printf("[SERVER] Job queued: socket=%d stream=%d max_tokens=%d learn=%d mopd=%d forced=%s\n",
		conn->fd, job.stream, job.max_tokens, job.learn, job.mopd,
		job.forced_target ? "YES" : "no");

	/* Push to queue (non-blocking for main thread) */
	queue_push(srv->queue, job);

	/* Clean up parsed request (but not content, worker owns it) */
	chat_request_free(&req);

	/* Mark connection as handed off - don't close in main loop */
	conn->fd = -1;
	return (1);
}

/*
** ============================================================================
** MOPD DISTILLATION ENDPOINT
** ============================================================================
** POST /v1/distill - Knowledge distillation from teacher model
** Expects: { "teacher_logprobs": [...], "target_token": N, "alpha": 0.8 }
*/

/* Parse float field from JSON (minimal parser) */
static float	distill_parse_float(const char *json, const char *key)
{
	const char	*pos;

	pos = strstr(json, key);
	if (!pos)
		return (0.0f);
	while (*pos && *pos != ':')
		pos++;
	if (*pos)
		pos++;
	return (strtof(pos, NULL));
}

/* Parse int field from JSON */
static int	distill_parse_int(const char *json, const char *key)
{
	const char	*pos;

	pos = strstr(json, key);
	if (!pos)
		return (-1);
	while (*pos && *pos != ':')
		pos++;
	if (*pos)
		pos++;
	return ((int)strtol(pos, NULL, 10));
}

/* Parse string field from JSON into buffer */
static int	distill_parse_string(const char *json, const char *key,
				char *buf, int buf_size)
{
	const char	*pos;
	const char	*start;
	const char	*end;
	int			len;

	pos = strstr(json, key);
	if (!pos)
		return (0);
	pos = strchr(pos, ':');
	if (!pos)
		return (0);
	while (*pos && *pos != '"')
		pos++;
	if (!*pos)
		return (0);
	start = pos + 1;  /* Skip opening quote */
	end = start;
	while (*end && *end != '"')
	{
		if (*end == '\\' && *(end + 1))
			end++;  /* Skip escaped char */
		end++;
	}
	len = (int)(end - start);
	if (len >= buf_size)
		len = buf_size - 1;
	memcpy(buf, start, len);
	buf[len] = '\0';
	return (len);
}

/*
** ============================================================================
** PHASE 6: FLUID PERSISTENCE HANDLERS
** ============================================================================
*/

int	handle_fluid_save(t_server *srv, t_client_conn *conn)
{
	char	filename[256];
	char	filepath[512];
	int		ret;

	if (!conn->request.body || conn->request.body_len == 0)
	{
		send_error_response(conn, 400, "No request body");
		return (0);
	}
	
	/* Parse filename from JSON */
	if (distill_parse_string(conn->request.body, "\"filename\"",
		filename, sizeof(filename)) <= 0)
	{
		send_error_response(conn, 400, "Missing 'filename' field");
		return (0);
	}
	
	/* Security: prevent path traversal */
	if (strchr(filename, '/') || strchr(filename, '\\') || strstr(filename, ".."))
	{
		send_error_response(conn, 400, "Invalid filename (no paths allowed)");
		return (0);
	}
	
	/* Build full path */
	snprintf(filepath, sizeof(filepath), "skills/%s", filename);
	
	/* Save */
	ret = fluid_save(srv->engine, filepath);
	if (ret < 0)
	{
		send_error_response(conn, 500, "Failed to save fluid state");
		return (0);
	}
	
	printf("[SERVER] Fluid state saved to: %s\n", filepath);
	send_json_response(conn, 200,
		"{\"status\":\"saved\",\"filename\":\"" "skills/%s" "\"}");
	return (0);
}

int	handle_fluid_load(t_server *srv, t_client_conn *conn)
{
	char	filename[256];
	char	filepath[512];
	int		ret;

	if (!conn->request.body || conn->request.body_len == 0)
	{
		send_error_response(conn, 400, "No request body");
		return (0);
	}
	
	/* Parse filename from JSON */
	if (distill_parse_string(conn->request.body, "\"filename\"",
		filename, sizeof(filename)) <= 0)
	{
		send_error_response(conn, 400, "Missing 'filename' field");
		return (0);
	}
	
	/* Security: prevent path traversal */
	if (strchr(filename, '/') || strchr(filename, '\\') || strstr(filename, ".."))
	{
		send_error_response(conn, 400, "Invalid filename (no paths allowed)");
		return (0);
	}
	
	/* Build full path */
	snprintf(filepath, sizeof(filepath), "skills/%s", filename);
	
	/* Load */
	ret = fluid_load(srv->engine, filepath);
	if (ret < 0)
	{
		send_error_response(conn, 404, "Fluid file not found or invalid");
		return (0);
	}
	
	printf("[SERVER] Fluid state loaded from: %s\n", filepath);
	send_json_response(conn, 200, "{\"status\":\"loaded\"}");
	return (0);
}

int	handle_fluid_list(t_server *srv, t_client_conn *conn)
{
	(void)srv;
	/* Simple implementation: just return a message */
	/* TODO: Actually list files in skills/ directory */
	send_json_response(conn, 200,
		"{\"status\":\"ok\",\"message\":\"List skills/ directory for .fluid files\"}");
	return (0);
}

int	handle_distill(t_server *srv, t_client_conn *conn)
{
	t_sparse_prob	teacher_probs[32];  /* Stack alloc for speed */
	int				num_probs;
	float			alpha;
	int				target_token;
	const char		*body;
	const char		*cursor;
	const char		*obj_start;
	int				tid;
	float			logprob;
	char			token_str[256];

	if (!conn->request.body || conn->request.body_len == 0)
	{
		send_error_response(conn, 400, "No request body");
		return (0);
	}

	body = conn->request.body;
	alpha = distill_parse_float(body, "\"alpha\"");
	target_token = distill_parse_int(body, "\"target_token\"");

	/* Clamp alpha to [0, 1] */
	if (alpha < 0.0f)
		alpha = 0.0f;
	if (alpha > 1.0f)
		alpha = 1.0f;

	/* Parse sparse teacher_logprobs array */
	num_probs = 0;
	cursor = strstr(body, "\"teacher_logprobs\"");
	if (cursor)
	{
		cursor = strchr(cursor, '[');
		while (cursor && *cursor != ']' && num_probs < 32)
		{
			obj_start = strchr(cursor, '{');
			if (!obj_start)
				break ;
			
			/* Try string-based token first (MOPD Phase 4) */
			tid = -1;
			if (distill_parse_string(obj_start, "\"token_str\"", token_str, 256) > 0)
			{
				/* Use tokenizer to convert string -> ID */
				if (srv->tokenizer)
					tid = tokenizer_lookup_id(srv->tokenizer, token_str);
			}
			
			/* Fallback: numeric token field (backwards compat) */
			if (tid < 0)
				tid = distill_parse_int(obj_start, "\"token\"");
			
			logprob = distill_parse_float(obj_start, "\"logprob\"");
			
			/* Only add valid tokens within vocab range */
			if (tid >= 0 && srv->engine && tid < srv->engine->config.vocab_size)
			{
				teacher_probs[num_probs].token_id = tid;
				teacher_probs[num_probs].prob = expf(logprob);  /* Log -> Lin */
				num_probs++;
			}
			cursor = strchr(obj_start, '}');
			if (cursor)
				cursor++;
		}
	}

	/* Check if engine has valid logits from recent forward pass */
	if (!srv->engine || !srv->engine->state.logits)
	{
		send_error_response(conn, 400, "No active logits. Run inference first.");
		return (0);
	}

	/* Execute distillation backward pass */
	backward_step_distill(srv->engine, teacher_probs, num_probs,
		target_token, alpha, srv->ctx.session_pos);

	/*
	** PHASE 5: Teacher Forcing
	** After learning from current logits, advance to next position
	** by feeding the teacher's chosen token through forward pass.
	*/
	int advance_token = -1;
	char advance_str[256];
	
	/* Try string-based token first */
	if (distill_parse_string(body, "\"advance_with_token_str\"", advance_str, 256) > 0)
	{
		if (srv->tokenizer)
			advance_token = tokenizer_lookup_id(srv->tokenizer, advance_str);
	}
	
	/* Fallback: numeric token ID */
	if (advance_token < 0)
		advance_token = distill_parse_int(body, "\"advance_with_token\"");

	/* Execute forward pass to advance engine state */
	if (advance_token >= 0 && advance_token < srv->engine->config.vocab_size)
	{
		transformer_forward(srv->engine, advance_token, srv->ctx.session_pos);
		srv->ctx.session_pos++;
	}

	/* Send success response with position info */
	{
		char resp[256];
		snprintf(resp, sizeof(resp),
			"{\"status\":\"ok\",\"distilled\":true,"
			"\"num_teacher_probs\":%d,\"pos\":%d,\"advanced\":%s}",
			num_probs, srv->ctx.session_pos,
			(advance_token >= 0) ? "true" : "false");
		send_json_response(conn, 200, resp);
	}
	return (0);
}

/*
** ============================================================================
** REQUEST DISPATCH
** ============================================================================
** Phase 9: Flexible URL routing for OpenAI SDK compatibility
** Accepts: /v1/chat/completions, /api/v1/chat/completions, /chat/completions, etc.
*/

/* Helper: check if path ends with suffix */
static int	path_ends_with(const char *path, const char *suffix)
{
	size_t	path_len;
	size_t	suffix_len;

	path_len = strlen(path);
	suffix_len = strlen(suffix);
	if (path_len < suffix_len)
		return (0);
	return (strcmp(path + path_len - suffix_len, suffix) == 0);
}

/* Helper: check if path contains substring */
static int	path_contains(const char *path, const char *sub)
{
	return (strstr(path, sub) != NULL);
}

int	handle_request(t_server *srv, t_client_conn *conn)
{
	/* GET /health (exact or ends with) */
	if (strcmp(conn->request.method, "GET") == 0
		&& (strcmp(conn->request.path, "/health") == 0
			|| path_ends_with(conn->request.path, "/health")))
	{
		send_json_response(conn, 200, "{\"status\":\"ok\"}");
		return (0);
	}

	/* GET /models (flexible: /v1/models, /api/v4/models, etc.) */
	if (strcmp(conn->request.method, "GET") == 0
		&& path_ends_with(conn->request.path, "/models"))
	{
		send_json_response(conn, 200,
			"{\"object\":\"list\",\"data\":["
			"{\"id\":\"42-engine\",\"object\":\"model\",\"owned_by\":\"42berlin\"}"
			"]}");
		return (0);
	}

	/* POST /chat/completions (flexible: /v1/, /api/v4/, etc.) */
	if (strcmp(conn->request.method, "POST") == 0
		&& path_ends_with(conn->request.path, "/chat/completions"))
	{
		return (handle_chat_completions(srv, conn));
	}

	/* POST /completions (legacy OpenAI endpoint) */
	if (strcmp(conn->request.method, "POST") == 0
		&& path_ends_with(conn->request.path, "/completions")
		&& !path_contains(conn->request.path, "/chat/"))
	{
		/* Treat legacy completions as chat completions */
		return (handle_chat_completions(srv, conn));
	}

	/* POST /distill (MOPD: teacher distillation endpoint) */
	if (strcmp(conn->request.method, "POST") == 0
		&& path_ends_with(conn->request.path, "/distill"))
	{
		return (handle_distill(srv, conn));
	}

	/* POST /fluid/save (Phase 6: Persistence) */
	if (strcmp(conn->request.method, "POST") == 0
		&& path_ends_with(conn->request.path, "/fluid/save"))
	{
		return (handle_fluid_save(srv, conn));
	}

	/* POST /fluid/load (Phase 6: Persistence) */
	if (strcmp(conn->request.method, "POST") == 0
		&& path_ends_with(conn->request.path, "/fluid/load"))
	{
		return (handle_fluid_load(srv, conn));
	}

	/* GET /fluid/list (Phase 6: List available skills) */
	if (strcmp(conn->request.method, "GET") == 0
		&& path_ends_with(conn->request.path, "/fluid/list"))
	{
		return (handle_fluid_list(srv, conn));
	}

	/* 404 for everything else */
	send_error_response(conn, 404, "Not found");
	return (0);
}

/*
** ============================================================================
** SERVER LIFECYCLE
** ============================================================================
*/

int	server_init(t_server *srv, int port, t_transformer *engine,
		t_tokenizer *tok)
{
	struct sockaddr_in	addr;
	int					i;

	/* Save mtp pointer before zeroing (set by caller before server_init) */
	struct s_mtp_engine *saved_mtp = srv->mtp;

	memset(srv, 0, sizeof(*srv));
	srv->mtp = saved_mtp;  /* Restore MTP pointer */
	srv->port = port;
	srv->engine = engine;
	srv->tokenizer = tok;
	srv->max_conns = SERVER_MAX_CONNECTIONS;
	srv->running = 1;
	ctx_init(&srv->ctx);

	/* Create socket */
	srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (srv->listen_fd < 0)
	{
		perror("socket");
		return (-1);
	}
	set_reuseaddr(srv->listen_fd);
	set_nonblocking(srv->listen_fd);

	/* Bind */
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(port);
	if (bind(srv->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("bind");
		close(srv->listen_fd);
		return (-1);
	}

	/* Listen */
	if (listen(srv->listen_fd, 128) < 0)
	{
		perror("listen");
		close(srv->listen_fd);
		return (-1);
	}

#if USE_EPOLL
	/* Create epoll instance */
	srv->epoll_fd = epoll_create1(0);
	if (srv->epoll_fd < 0)
	{
		perror("epoll_create1");
		close(srv->listen_fd);
		return (-1);
	}
	/* Add listen socket to epoll */
	/* [PERF PATCH #9] Use ev.data.ptr for O(1) lookup
	** Listen socket uses NULL as marker (no t_client_conn for it)
	** Client sockets will store pointer to their t_client_conn */
	{
		struct epoll_event	ev;

		ev.events = EPOLLIN;
		ev.data.ptr = NULL;  /* NULL = listen socket marker */
		epoll_ctl(srv->epoll_fd, EPOLL_CTL_ADD, srv->listen_fd, &ev);
	}
#endif

	/* Allocate connection pool */
	srv->conns = xcalloc(srv->max_conns, sizeof(t_client_conn));
	i = 0;
	while (i < srv->max_conns)
	{
		srv->conns[i].fd = -1;
		i++;
	}

	/* Phase 8: Initialize job queue and worker thread */
	srv->queue = queue_init(QUEUE_DEFAULT_CAPACITY);
	if (!srv->queue)
	{
		fprintf(stderr, "Failed to create job queue\n");
		close(srv->listen_fd);
		return (-1);
	}
	srv->worker_ctx = xcalloc(1, sizeof(t_worker_ctx));
	srv->worker_ctx->queue = srv->queue;
	srv->worker_ctx->engine = engine;
	srv->worker_ctx->tokenizer = tok;
	srv->worker_ctx->mtp = (t_mtp_engine *)srv->mtp;  /* Wire MTP for burst gen */
	/* Disabled for production:
	printf("[DEBUG] server.c: srv->mtp = %p, worker_ctx->mtp = %p\n",
		(void *)srv->mtp, (void *)srv->worker_ctx->mtp);
	*/
	if (worker_start(srv->worker_ctx, &srv->worker_thread) != 0)
	{
		fprintf(stderr, "Failed to start worker thread\n");
		queue_free(srv->queue);
		close(srv->listen_fd);
		return (-1);
	}

	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║           42-BERLIN-ENGINE DAEMON v0.2 (Async)               ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");
	printf("  Listening on: http://0.0.0.0:%d\n", port);
	printf("  Mode: Async (worker thread + SSE streaming)\n");
	printf("  Endpoints:\n");
	printf("    POST /v1/chat/completions - Chat completion (streaming)\n");
	printf("    GET  /v1/models           - List models\n");
	printf("    GET  /health              - Health check\n");
	printf("\n");

	return (0);
}

#if USE_EPOLL

int	server_run(t_server *srv)
{
	struct epoll_event	events[64];
	int					n_events;
	int					i;

	while (srv->running)
	{
		n_events = epoll_wait(srv->epoll_fd, events, 64, 1000);
		if (n_events < 0)
		{
			if (errno == EINTR)
				continue ;
			perror("epoll_wait");
			break ;
		}

	i = 0;
		while (i < n_events)
		{
			t_client_conn	*conn;

			/* [PERF PATCH #9] O(1) connection lookup via ev.data.ptr
			** Old code: loop through all conns to find matching fd (O(N))
			** New code: direct pointer access (O(1))
			** NULL = listen socket (accept new connection) */
			conn = (t_client_conn *)events[i].data.ptr;
			
			/* New connection (listen socket has NULL marker) */
			if (conn == NULL)
			{
				struct sockaddr_in	client_addr;
				socklen_t			addr_len;
				int					client_fd;
				t_client_conn		*new_conn;
				struct epoll_event	ev;

				addr_len = sizeof(client_addr);
				client_fd = accept(srv->listen_fd,
					(struct sockaddr *)&client_addr, &addr_len);
				if (client_fd >= 0)
				{
					new_conn = find_free_conn(srv);
					if (new_conn)
					{
						set_nonblocking(client_fd);
						set_tcp_nodelay(client_fd);
						conn_init(new_conn, client_fd);
						/* [PERF PATCH #9] Store conn pointer for O(1) lookup */
						ev.events = EPOLLIN | EPOLLET;
						ev.data.ptr = new_conn;  /* Direct pointer! */
						epoll_ctl(srv->epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
						srv->n_conns++;
					}
					else
					{
						close(client_fd);  /* No free slots */
					}
				}
			}
			/* Client data - O(1) direct access! */
			else
			{
				int		result;
				int		fd = conn->fd;  /* Get fd from connection struct */

				/* Handle based on state */
				if (events[i].events & EPOLLIN)
				{
					ssize_t	n;

					n = read(fd, conn->read_buf + conn->read_len,
						SERVER_BUFFER_SIZE - conn->read_len - 1);
					if (n <= 0)
					{
						/* Connection closed or error */
						epoll_ctl(srv->epoll_fd, EPOLL_CTL_DEL, fd, NULL);
						conn_reset(conn);
						srv->n_conns--;
					}
					else
					{
						conn->read_len += n;
						conn->read_buf[conn->read_len] = '\0';
						
						/* Try to parse request */
						result = parse_http_request(conn);
						if (result > 0)
						{
							/* Request complete - process it */
							handle_request(srv, conn);
							
							/* Immediately try to send response */
							if (conn->state == CONN_STATE_WRITING && conn->write_buf)
							{
								ssize_t sent;
								sent = write(fd, conn->write_buf + conn->write_pos,
									conn->write_len - conn->write_pos);
								if (sent > 0)
								{
									conn->write_pos += sent;
									if (conn->write_pos >= conn->write_len)
									{
										/* Response complete - close or reset */
										epoll_ctl(srv->epoll_fd, EPOLL_CTL_DEL, fd, NULL);
										conn_reset(conn);
										srv->n_conns--;
									}
								}
							}
						}
						else if (result < 0)
						{
							/* Parse error */
							send_error_response(conn, 400, "Malformed request");
						}
					}
				}
				if ((events[i].events & EPOLLOUT)
					&& conn->state == CONN_STATE_WRITING)
				{
					ssize_t	sent;

					sent = write(fd, conn->write_buf + conn->write_pos,
						conn->write_len - conn->write_pos);
					if (sent > 0)
					{
						conn->write_pos += sent;
						if (conn->write_pos >= conn->write_len)
						{
							/* Response complete */
							if (!conn->request.keep_alive)
							{
								epoll_ctl(srv->epoll_fd, EPOLL_CTL_DEL, fd, NULL);
								conn_reset(conn);
								srv->n_conns--;
							}
							else
							{
								/* Reset for next request */
								if (conn->write_buf)
									free(conn->write_buf);
								conn->write_buf = NULL;
								if (conn->request.body)
									free(conn->request.body);
								conn->request.body = NULL;
								conn->read_len = 0;
								conn->write_len = 0;
								conn->write_pos = 0;
								conn->state = CONN_STATE_READING;
							}
						}
					}
				}
			}
			i++;
		}
	}
	return (0);
}

#else /* USE_EPOLL */

/* Fallback select() implementation for non-Linux */
int	server_run(t_server *srv)
{
	fd_set			read_fds;
	fd_set			write_fds;
	int				max_fd;
	struct timeval	tv;

	while (srv->running)
	{
		FD_ZERO(&read_fds);
		FD_ZERO(&write_fds);
		FD_SET(srv->listen_fd, &read_fds);
		max_fd = srv->listen_fd;

		/* Add client connections */
		for (int i = 0; i < srv->max_conns; i++)
		{
			if (srv->conns[i].fd >= 0)
			{
				if (srv->conns[i].state == CONN_STATE_READING)
					FD_SET(srv->conns[i].fd, &read_fds);
				if (srv->conns[i].state == CONN_STATE_WRITING)
					FD_SET(srv->conns[i].fd, &write_fds);
				if (srv->conns[i].fd > max_fd)
					max_fd = srv->conns[i].fd;
			}
		}

		tv.tv_sec = 1;
		tv.tv_usec = 0;
		if (select(max_fd + 1, &read_fds, &write_fds, NULL, &tv) < 0)
		{
			if (errno == EINTR)
				continue ;
			break ;
		}

		/* Handle new connections */
		if (FD_ISSET(srv->listen_fd, &read_fds))
		{
			int	client_fd = accept(srv->listen_fd, NULL, NULL);
			if (client_fd >= 0)
			{
				t_client_conn *conn = find_free_conn(srv);
				if (conn)
				{
					set_nonblocking(client_fd);
					conn_init(conn, client_fd);
					srv->n_conns++;
				}
				else
					close(client_fd);
			}
		}

		/* Handle client I/O (simplified) */
		for (int i = 0; i < srv->max_conns; i++)
		{
			if (srv->conns[i].fd < 0)
				continue ;
			/* ... similar to epoll handler ... */
		}
	}
	return (0);
}

#endif /* USE_EPOLL */

void	server_shutdown(t_server *srv)
{
	int	i;

	srv->running = 0;
	printf("\nShutting down server...\n");

	/* Stop worker thread first */
	if (srv->worker_ctx)
	{
		worker_stop(srv->worker_ctx, srv->worker_thread);
		free(srv->worker_ctx);
		srv->worker_ctx = NULL;
	}
	if (srv->queue)
	{
		queue_free(srv->queue);
		srv->queue = NULL;
	}

	/* Close all connections */
	i = 0;
	while (i < srv->max_conns)
	{
		if (srv->conns[i].fd >= 0)
			conn_reset(&srv->conns[i]);
		i++;
	}
	free(srv->conns);

#if USE_EPOLL
	close(srv->epoll_fd);
#endif
	close(srv->listen_fd);
	printf("Server stopped.\n");
}
