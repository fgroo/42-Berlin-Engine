/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   teacher.h                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/24 01:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/24 01:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TEACHER_H
# define TEACHER_H

/*
** ============================================================================
** MOPD TEACHER CLIENT API
** ============================================================================
** Knowledge distillation from external teacher model (GLM-4.7).
** ============================================================================
*/

/*
** Fetch a completion from the teacher model.
** @param prompt: The user prompt
** @param api_key: API key for authentication
** @param max_tokens: Maximum tokens to generate (50 recommended for speed)
** @return: Heap-allocated response string (caller must free), or NULL on error
*/
char	*teacher_fetch_completion(const char *prompt, const char *api_key,
								  int max_tokens);

/*
** Test the connection to the teacher API.
** @param api_key: API key for authentication
** @return: 1 on success, 0 on failure
*/
int		teacher_test_connection(const char *api_key);

/*
** Learning mode flags for worker
*/
typedef enum e_learn_mode
{
	LEARN_NONE = 0,      /* No learning */
	LEARN_SELF = 1,      /* Self-correction (learn from own samples) */
	LEARN_MOPD = 2       /* MOPD (learn from teacher samples) */
}	t_learn_mode;

#endif
