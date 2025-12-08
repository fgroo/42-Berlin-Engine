/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   loader.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/08 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "loader.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
** VISION ENCODER LOBOTOMY: Skip multimodal tensors
** Ministral 3B is native multimodal - vision encoder weights corrupt
** text model if loaded. Skip tensors containing these substrings.
*/
static int	is_vision_tensor(const char *name)
{
	if (strstr(name, "vision"))
		return (1);
	if (strstr(name, "image"))
		return (1);
	if (strstr(name, "patch"))
		return (1);
	if (strstr(name, "pre_mm"))
		return (1);
	if (strstr(name, "adapter"))
		return (1);
	return (0);
}

int	extract_name(t_model *model, const char *qs, const char *qe)
{
	int	len;
	int	idx;

	len = qe - (qs + 1);
	if (len >= MAX_NAME_LEN)
		len = MAX_NAME_LEN - 1;
	idx = model->num_tensors;
	strncpy(model->tensors[idx].name, qs + 1, len);
	model->tensors[idx].name[len] = '\0';
	/* Skip metadata */
	if (strcmp(model->tensors[idx].name, "__metadata__") == 0)
		return (0);
	/* Skip vision encoder tensors (multimodal lobotomy) */
	if (is_vision_tensor(model->tensors[idx].name))
	{
		printf("[SKIP] Vision tensor: %s\n", model->tensors[idx].name);
		return (0);
	}
	return (1);
}

static void	parse_digit(t_model *model, const char **d, int *dim)
{
	int	idx;

	idx = model->num_tensors;
	model->tensors[idx].tensor.shape[*dim] = atoi(*d);
	(*dim)++;
	while (**d >= '0' && **d <= '9')
		(*d)++;
}

void	parse_shape(t_model *model, const char *obj_start)
{
	const char	*arr;
	const char	*d;
	int			dim;
	int			idx;

	arr = strstr(obj_start, "\"shape\"");
	if (!arr)
		return ;
	arr = strchr(arr, '[');
	if (!arr)
		return ;
	dim = 0;
	d = arr + 1;
	idx = model->num_tensors;
	while (*d && *d != ']')
	{
		if (*d >= '0' && *d <= '9')
			parse_digit(model, &d, &dim);
		else
			d++;
	}
	model->tensors[idx].tensor.ndim = dim;
}
