/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   server.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 21:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 21:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef SERVER_H
# define SERVER_H

# include "inference/inference.h"
# include "tokenizer/tokenizer.h"
# include "engine_context.h"
# include <stdint.h>
# include <pthread.h>

/*
** ============================================================================
** 42-BERLIN-ENGINE HTTP DAEMON
** ============================================================================
** OpenAI-compatible API server for LLM inference.
** Single-threaded event loop using epoll (Linux) or select (fallback).
**
** Endpoints:
**   POST /v1/chat/completions - Chat completion (streaming/non-streaming)
**   GET  /v1/models           - List available models
**   GET  /health              - Health check
**
** Protocol:
**   HTTP/1.1 with JSON request/response bodies.
**   Streaming uses Server-Sent Events (SSE).
** ============================================================================
*/

/* Default configuration */
# define SERVER_DEFAULT_PORT     8080
# define SERVER_MAX_CONNECTIONS  64
# define SERVER_BUFFER_SIZE      65536
# define SERVER_MAX_TOKENS       4096

/* [SECURITY] Request size limits - prevent DoS and buffer overflow attacks */
# define MAX_BODY_SIZE           (1024 * 1024 * 10)  /* 10 MB max request body */
# define MAX_HEADER_SIZE         8192                 /* 8 KB max headers */

/* Connection states */
typedef enum e_conn_state
{
	CONN_STATE_READING,        /* Reading HTTP request */
	CONN_STATE_PROCESSING,     /* Generating response */
	CONN_STATE_WRITING,        /* Sending HTTP response */
	CONN_STATE_STREAMING,      /* Streaming SSE events */
	CONN_STATE_CLOSING         /* Connection closing */
}	t_conn_state;

/* HTTP request (parsed) */
typedef struct s_http_request
{
	char		method[16];      /* GET, POST, etc. */
	char		path[256];       /* /v1/chat/completions */
	char		*body;           /* JSON body (heap allocated) */
	int			body_len;
	int			content_length;  /* Content-Length header value */
	int			keep_alive;      /* Connection: keep-alive */
	int			stream;          /* "stream": true in request */
}	t_http_request;

/* Client connection */
typedef struct s_client_conn
{
	int				fd;              /* Socket file descriptor */
	t_conn_state	state;
	char			read_buf[SERVER_BUFFER_SIZE];
	int				read_len;
	char			*write_buf;      /* Response buffer (heap) */
	int				write_len;
	int				write_pos;       /* Bytes already sent */
	t_http_request	request;
	int				gen_pos;         /* Current generation position */
	int				*tokens;         /* Token buffer for generation */
	int				n_tokens;
}	t_client_conn;

/* Server context */
typedef struct s_server
{
	int				listen_fd;       /* Listening socket */
	int				port;
	int				epoll_fd;        /* epoll instance (Linux) */
	t_client_conn	*conns;          /* Connection pool */
	int				n_conns;
	int				max_conns;
	t_transformer	*engine;         /* Inference engine */
	t_tokenizer		*tokenizer;
	t_engine_context ctx;            /* Thread-safe session state */
	volatile int	running;         /* Server loop control */
	/* Phase 8: Async worker */
	struct s_job_queue		*queue;          /* Job queue */
	struct s_worker_ctx		*worker_ctx;     /* Worker context */
	pthread_t				worker_thread;   /* Worker thread handle */
}	t_server;

/* Server lifecycle */
int		server_init(t_server *srv, int port, t_transformer *engine,
			t_tokenizer *tok);
int		server_run(t_server *srv);
void	server_shutdown(t_server *srv);

/* Request handling */
int		handle_request(t_server *srv, t_client_conn *conn);
int		parse_http_request(t_client_conn *conn);

/* Response helpers */
void	send_json_response(t_client_conn *conn, int status,
			const char *json);
void	send_sse_event(t_client_conn *conn, const char *event,
			const char *data);
void	send_error_response(t_client_conn *conn, int status,
			const char *message);

#endif
