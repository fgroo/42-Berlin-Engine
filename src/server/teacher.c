/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   teacher.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/24 01:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/24 01:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ============================================================================
** MOPD TEACHER CLIENT - GLM-4.7 Integration
** ============================================================================
** Fetches "gold standard" completions from external teacher model (GLM-4.7).
** The student (Ministral) learns to mimic the teacher's outputs during
** generation via Hard Label Distillation.
**
** Implementation uses popen() + curl binary to avoid libcurl dependency.
** This is slower but works without additional system packages.
** ============================================================================
*/

#include "teacher.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEACHER_RESPONSE_MAX 16384
#define TEACHER_TIMEOUT_SEC 5

/*
** Extract string value after a key in JSON.
** Helper for extract_content and extract_reasoning_content.
*/
static char	*extract_json_string(const char *json, const char *key)
{
	const char	*start;
	const char	*end;
	const char	*key_pos;
	char		*result;
	size_t		len;
	char		search_key[64];

	snprintf(search_key, sizeof(search_key), "\"%s\":", key);
	key_pos = strstr(json, search_key);
	if (!key_pos)
		return (NULL);
	
	/* Skip to value (find opening quote) */
	start = strchr(key_pos + strlen(search_key), '"');
	if (!start)
		return (NULL);
	start++;  /* Skip opening quote */
	
	/* Find closing quote (handle escapes) */
	end = start;
	while (*end && *end != '"')
	{
		if (*end == '\\' && *(end + 1))
			end += 2;  /* Skip escaped char */
		else
			end++;
	}
	
	len = end - start;
	if (len == 0)
		return (NULL);  /* Empty string - treat as not found */
	
	result = malloc(len + 1);
	if (!result)
		return (NULL);
	
	/* Copy with unescape */
	{
		const char	*src = start;
		char		*dst = result;
		
		while (src < end)
		{
			if (*src == '\\' && src + 1 < end)
			{
				src++;
				if (*src == 'n')
					*dst++ = '\n';
				else if (*src == 'r')
					*dst++ = '\r';
				else if (*src == 't')
					*dst++ = '\t';
				else if (*src == '\\')
					*dst++ = '\\';
				else if (*src == '"')
					*dst++ = '"';
				else
					*dst++ = *src;
				src++;
			}
			else
			{
				*dst++ = *src++;
			}
		}
		*dst = '\0';
	}
	
	return (result);
}

/*
** Parse "content" or "reasoning_content" field from OpenAI/GLM-style JSON.
** GLM-4.7 is a thinking model that puts actual response in reasoning_content.
*/
static char	*extract_content(const char *json)
{
	char	*content;

	/* Try normal content first */
	content = extract_json_string(json, "content");
	if (content)
		return (content);
	
	/* GLM-4.7 uses reasoning_content for the actual response */
	content = extract_json_string(json, "reasoning_content");
	if (content)
		return (content);

	return (NULL);
}

/*
** Escape string for JSON (minimal escaping)
*/
static void	json_escape_str(const char *src, char *dst, size_t max)
{
	size_t	i;
	size_t	j;

	i = 0;
	j = 0;
	while (src[i] && j < max - 2)
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

/*
** Fetch teacher completion using curl binary via popen().
** Returns heap-allocated string (caller must free), or NULL on error.
*/
char	*teacher_fetch_completion(const char *prompt, const char *api_key,
								  int max_tokens)
{
	char		cmd[8192];
	char		escaped_prompt[4096];
	char		*response;
	char		*content;
	FILE		*fp;
	size_t		total_read;
	size_t		chunk;

	if (!prompt || !api_key)
		return (NULL);

	/* Escape prompt for JSON */
	json_escape_str(prompt, escaped_prompt, sizeof(escaped_prompt) - 1);

	/* Build curl command */
	snprintf(cmd, sizeof(cmd),
		"curl -s --max-time %d "
		"-X POST 'https://api.z.ai/api/coding/paas/v4/chat/completions' "
		"-H 'Content-Type: application/json' "
		"-H 'Authorization: Bearer %s' "
		"-d '{"
			"\"model\": \"glm-4.7\","
			"\"messages\": [{\"role\": \"user\", \"content\": \"%s\"}],"
			"\"max_tokens\": %d,"
			"\"stream\": false"
		"}'",
		TEACHER_TIMEOUT_SEC,
		api_key,
		escaped_prompt,
		max_tokens > 0 ? max_tokens : 50);

	/* Execute curl */
	fp = popen(cmd, "r");
	if (!fp)
	{
		fprintf(stderr, "[TEACHER] Failed to execute curl\n");
		return (NULL);
	}

	/* Read response */
	response = malloc(TEACHER_RESPONSE_MAX);
	if (!response)
	{
		pclose(fp);
		return (NULL);
	}
	
	total_read = 0;
	while ((chunk = fread(response + total_read, 1, 
			TEACHER_RESPONSE_MAX - total_read - 1, fp)) > 0)
	{
		total_read += chunk;
	}
	response[total_read] = '\0';
	pclose(fp);

	/* Check for empty response */
	if (total_read == 0)
	{
		fprintf(stderr, "[TEACHER] Empty response (timeout?)\n");
		free(response);
		return (NULL);
	}

	/* Check for API error */
	if (strstr(response, "\"error\""))
	{
		fprintf(stderr, "[TEACHER] API error: %.200s\n", response);
		free(response);
		return (NULL);
	}

	/* Extract content from response */
	content = extract_content(response);
	free(response);

	if (!content)
	{
		fprintf(stderr, "[TEACHER] Failed to parse response\n");
		return (NULL);
	}

	return (content);
}

/*
** Test the teacher connection.
** Returns 1 on success, 0 on failure.
*/
int	teacher_test_connection(const char *api_key)
{
	char	*response;

	printf("[TEACHER] Testing connection to GLM-4.7...\n");
	response = teacher_fetch_completion("Say just 'OK'", api_key, 5);
	if (!response)
	{
		printf("[TEACHER] ❌ Connection failed\n");
		return (0);
	}
	printf("[TEACHER] ✓ Connection OK. Response: '%.50s'\n", response);
	free(response);
	return (1);
}
