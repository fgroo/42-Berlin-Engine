/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   json_parse.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/20 00:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/20 00:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "json_parse.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

/*
** ============================================================================
** JSON HELPERS
** ============================================================================
*/

const char	*json_skip_ws(const char *s)
{
	while (*s && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r'))
		s++;
	return (s);
}

/*
** Skip a complete JSON value (recursive for objects/arrays)
*/
const char	*json_skip_value(const char *s)
{
	int	depth;

	s = json_skip_ws(s);
	if (*s == '"')
	{
		/* String: skip to closing quote */
		s++;
		while (*s && *s != '"')
		{
			if (*s == '\\' && *(s + 1))
				s++;
			s++;
		}
		if (*s == '"')
			s++;
	}
	else if (*s == '{' || *s == '[')
	{
		/* Object or array: track nesting depth */
		char	open = *s;
		char	close = (open == '{') ? '}' : ']';

		depth = 1;
		s++;
		while (*s && depth > 0)
		{
			if (*s == '"')
			{
				s++;
				while (*s && *s != '"')
				{
					if (*s == '\\' && *(s + 1))
						s++;
					s++;
				}
				if (*s == '"')
					s++;
			}
			else
			{
				if (*s == open)
					depth++;
				else if (*s == close)
					depth--;
				s++;
			}
		}
	}
	else if (strncmp(s, "true", 4) == 0)
		s += 4;
	else if (strncmp(s, "false", 5) == 0)
		s += 5;
	else if (strncmp(s, "null", 4) == 0)
		s += 4;
	else
	{
		/* Number: skip digits, dots, e, signs */
		if (*s == '-')
			s++;
		while (*s && (isdigit((unsigned char)*s) || *s == '.' || *s == 'e'
				|| *s == 'E' || *s == '+' || *s == '-'))
			s++;
	}
	return (s);
}

/*
** Extract string value (returns heap-allocated, caller frees)
*/
char	*json_extract_string(const char *s, const char **end)
{
	char	*result;
	char	*dst;
	int		len;
	const char *start;

	if (*s != '"')
	{
		*end = s;
		return (NULL);
	}
	s++;
	start = s;
	/* First pass: count length */
	len = 0;
	while (*s && *s != '"')
	{
		if (*s == '\\' && *(s + 1))
		{
			s += 2;
			len++;
		}
		else
		{
			s++;
			len++;
		}
	}
	/* Allocate and copy with unescape */
	result = (char *)malloc(len + 1);
	if (!result)
	{
		*end = s;
		return (NULL);
	}
	s = start;
	dst = result;
	while (*s && *s != '"')
	{
		if (*s == '\\' && *(s + 1))
		{
			s++;
			if (*s == 'n')
				*dst++ = '\n';
			else if (*s == 'r')
				*dst++ = '\r';
			else if (*s == 't')
				*dst++ = '\t';
			else if (*s == '\\')
				*dst++ = '\\';
			else if (*s == '"')
				*dst++ = '"';
			else
				*dst++ = *s;
			s++;
		}
		else
		{
			*dst++ = *s++;
		}
	}
	*dst = '\0';
	if (*s == '"')
		s++;
	*end = s;
	return (result);
}

double	json_extract_number(const char *s, const char **end)
{
	double	result;
	char	*endptr;

	result = strtod(s, &endptr);
	*end = endptr;
	return (result);
}

int	json_extract_bool(const char *s, const char **end)
{
	if (strncmp(s, "true", 4) == 0)
	{
		*end = s + 4;
		return (1);
	}
	if (strncmp(s, "false", 5) == 0)
	{
		*end = s + 5;
		return (0);
	}
	*end = s;
	return (0);
}

/*
** ============================================================================
** CHAT REQUEST PARSING
** ============================================================================
*/

void	chat_request_init(t_chat_request *req)
{
	memset(req, 0, sizeof(*req));
	req->max_tokens = 256;
	req->temperature = 0.7f;
	req->top_p = 0.9f;
	req->enable_thinking = 0;
	req->thinking_budget = 0;
	req->force_response = NULL;  /* JSONL Teacher forcing */
}

void	chat_request_free(t_chat_request *req)
{
	if (req->model)
		free(req->model);
	if (req->content)
		free(req->content);
	if (req->force_response)
		free(req->force_response);
	req->model = NULL;
	req->content = NULL;
	req->force_response = NULL;
}

/*
** Find a key in current object level
** Returns pointer to value after colon, or NULL if not found
*/
static const char	*find_key(const char *json, const char *key)
{
	char	search[128];
	const char *p;

	snprintf(search, sizeof(search), "\"%s\"", key);
	p = json;
	while ((p = strstr(p, search)) != NULL)
	{
		p += strlen(search);
		p = json_skip_ws(p);
		if (*p == ':')
		{
			p++;
			return (json_skip_ws(p));
		}
	}
	return (NULL);
}

/*
** Extract last "content" from messages array
*/
static char	*extract_last_content(const char *json)
{
	const char	*p;
	const char	*last_content;
	char		*result;

	last_content = NULL;
	p = json;
	/* Find all "content": occurrences, keep last one */
	while ((p = strstr(p, "\"content\"")) != NULL)
	{
		p += 9;
		p = json_skip_ws(p);
		if (*p == ':')
		{
			p++;
			p = json_skip_ws(p);
			if (*p == '"')
				last_content = p;
		}
	}
	if (!last_content)
		return (NULL);
	result = json_extract_string(last_content, &p);
	return (result);
}

/*
** Parse thinking object: {"type": "enabled", "budget_tokens": 10000}
*/
static void	parse_thinking(const char *json, t_chat_request *req)
{
	const char	*p;
	const char	*end;
	char		*type_str;

	p = find_key(json, "type");
	if (p)
	{
		type_str = json_extract_string(p, &end);
		if (type_str)
		{
			if (strcmp(type_str, "enabled") == 0)
				req->enable_thinking = 1;
			free(type_str);
		}
	}
	p = find_key(json, "budget_tokens");
	if (p)
		req->thinking_budget = (int)json_extract_number(p, &end);
}

/*
** Main parser: extract fields from OpenAI-style request
** Tolerant: ignores unknown fields
*/
int	chat_request_parse(const char *json, t_chat_request *req)
{
	const char	*p;
	const char	*end;

	chat_request_init(req);
	if (!json)
		return (-1);

	/* model (optional) */
	p = find_key(json, "model");
	if (p && *p == '"')
		req->model = json_extract_string(p, &end);

	/* stream */
	p = find_key(json, "stream");
	if (p)
		req->stream = json_extract_bool(p, &end);

	/* max_tokens */
	p = find_key(json, "max_tokens");
	if (p)
		req->max_tokens = (int)json_extract_number(p, &end);

	/* temperature */
	p = find_key(json, "temperature");
	if (p)
		req->temperature = (float)json_extract_number(p, &end);

	/* top_p */
	p = find_key(json, "top_p");
	if (p)
		req->top_p = (float)json_extract_number(p, &end);

	/* thinking (DeepSeek R1 style) */
	p = find_key(json, "thinking");
	if (p && *p == '{')
		parse_thinking(p, req);

	/* learn (Phase 10: Runtime Learning / TTT) */
	p = find_key(json, "learn");
	if (p)
		req->learn = json_extract_bool(p, &end);

	/* mopd (Phase 10: MOPD - learn from teacher) */
	p = find_key(json, "mopd");
	if (p)
		req->mopd = json_extract_bool(p, &end);

	/* force_response: JSONL Teacher forcing (Dataset is the Boss) */
	p = find_key(json, "force_response");
	if (p && *p == '"')
	{
		char *start = (char *)p + 1;
		char *endq = start;
		while (*endq && *endq != '"')
		{
			if (*endq == '\\' && *(endq + 1))
				endq++;  /* Skip escaped char */
			endq++;
		}
		size_t len = endq - start;
		req->force_response = malloc(len + 1);
		if (req->force_response)
		{
			memcpy(req->force_response, start, len);
			req->force_response[len] = '\0';
		}
	}

	/* messages -> extract last content */
	req->content = extract_last_content(json);
	if (!req->content)
		return (-1);  /* No content found - error */

	return (0);
}
