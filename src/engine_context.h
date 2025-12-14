/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   engine_context.h                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/14 15:00:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/14 15:00:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef ENGINE_CONTEXT_H
# define ENGINE_CONTEXT_H

# include <stddef.h>

/*
** Engine Context: Encapsulates all session-specific state.
** NO GLOBALS. Thread-safe by design.
*/

# define CTX_UTF8_BUF_SIZE 256
# define CTX_RESPONSE_BUF_SIZE 2048

typedef struct s_engine_context
{
	/* Session position for RoPE continuity across turns */
	int				session_pos;
	
	/* UTF-8 accumulation buffer for multi-byte character handling */
	unsigned char	utf8_buf[CTX_UTF8_BUF_SIZE];
	int				utf8_len;
	
	/* Response buffer for stop string detection */
	char			response_buf[CTX_RESPONSE_BUF_SIZE];
	int				response_len;
	
	/* Nested learning step counter (per-session, not global) */
	int				nl_step;
	int				nl_skipped;
	int				nl_learn_steps;
}	t_engine_context;

/*
** Initialize context to safe defaults
*/
static inline void	ctx_init(t_engine_context *ctx)
{
	ctx->session_pos = 0;
	ctx->utf8_len = 0;
	ctx->response_len = 0;
	ctx->nl_step = 0;
	ctx->nl_skipped = 0;
	ctx->nl_learn_steps = 0;
}

/*
** Reset context for new conversation (preserves session)
*/
static inline void	ctx_reset_conversation(t_engine_context *ctx)
{
	ctx->session_pos = 0;
	ctx->utf8_len = 0;
	ctx->response_len = 0;
	ctx->nl_step = 0;
	ctx->nl_skipped = 0;
	ctx->nl_learn_steps = 0;
}

/*
** Reset UTF-8 buffer (call after flush)
*/
static inline void	ctx_reset_utf8(t_engine_context *ctx)
{
	ctx->utf8_len = 0;
}

/*
** Reset response buffer (call after each generation)
*/
static inline void	ctx_reset_response(t_engine_context *ctx)
{
	ctx->response_buf[0] = '\0';
	ctx->response_len = 0;
}

#endif
