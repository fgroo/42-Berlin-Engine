/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   json_parse.h                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/20 00:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/20 00:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef JSON_PARSE_H
# define JSON_PARSE_H

/*
** ============================================================================
** ROBUST JSON PARSER (Phase 9)
** ============================================================================
** Tolerant parser for OpenAI-compatible requests.
** - Ignores unknown fields (thinking, extra_body, etc.)
** - Handles nested objects/arrays gracefully
** - Extracts only what we need
**
** This is NOT a full JSON parser. It's a pragmatic extractor.
** ============================================================================
*/

/* Parsed chat completion request */
typedef struct s_chat_request
{
	char	*model;           /* Optional: model name */
	char	*content;         /* Last user message content */
	char	*force_response;  /* JSONL Teacher: forced target text */
	int		stream;           /* stream: true/false */
	int		max_tokens;       /* max_tokens (default: 256) */
	float	temperature;      /* temperature (default: 0.7) */
	float	top_p;            /* top_p (default: 0.9) */
	int		enable_thinking;  /* thinking.type == "enabled" */
	int		thinking_budget;  /* thinking.budget_tokens */
	int		learn;            /* learn: true = enable runtime learning */
	int		mopd;             /* mopd: true = use teacher for distillation */
}	t_chat_request;

/*
** Initialize with defaults
*/
void	chat_request_init(t_chat_request *req);

/*
** Parse JSON body into chat_request
** Returns 0 on success, -1 on error
** Tolerant: unknown fields are ignored
*/
int		chat_request_parse(const char *json, t_chat_request *req);

/*
** Free allocated strings in request
*/
void	chat_request_free(t_chat_request *req);

/*
** JSON helper: skip whitespace
*/
const char	*json_skip_ws(const char *s);

/*
** JSON helper: skip a value (string, number, object, array, bool, null)
** Returns pointer after the value
*/
const char	*json_skip_value(const char *s);

/*
** JSON helper: extract string value (caller must free)
** Expects s to point to opening quote
*/
char	*json_extract_string(const char *s, const char **end);

/*
** JSON helper: extract number
*/
double	json_extract_number(const char *s, const char **end);

/*
** JSON helper: extract boolean
*/
int		json_extract_bool(const char *s, const char **end);

#endif
