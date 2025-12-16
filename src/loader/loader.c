/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   loader.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/12 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "loader.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
** Categorize tensor by name pattern for memory management:
** - VISION: Lazy loaded, excluded from sparse attention in text-only mode
** - FLUID:  Copied to arena for gradient updates (LoRA adapters)
** - FROZEN: Base weights, mmap read-only
*/
static t_tensor_category	categorize_tensor(const char *name)
{
	if (strstr(name, "vision") || strstr(name, "image") ||
		strstr(name, "patch") || strstr(name, "pre_mm") ||
		strstr(name, "adapter") || strstr(name, "vit") ||
		strstr(name, "projector"))
		return (TENSOR_VISION);
	if (strstr(name, "lora") || strstr(name, "fluid"))
		return (TENSOR_FLUID);
	return (TENSOR_FROZEN);
}

int	extract_name(t_model *model, const char *qs, const char *qe)
{
	int					len;
	int					idx;
	t_tensor_category	cat;

	len = qe - (qs + 1);
	if (len >= MAX_NAME_LEN)
		len = MAX_NAME_LEN - 1;
	idx = model->num_tensors;
	strncpy(model->tensors[idx].name, qs + 1, len);
	model->tensors[idx].name[len] = '\0';
	/* Skip metadata only */
	if (strcmp(model->tensors[idx].name, "__metadata__") == 0)
		return (0);
	/* Categorize tensor (don't skip vision anymore - lazy loading!) */
	cat = categorize_tensor(model->tensors[idx].name);
	model->tensors[idx].category = cat;
	if (cat == TENSOR_VISION)
	{
		model->num_vision_tensors++;
		printf("[LAZY] Vision tensor: %s\n", model->tensors[idx].name);
	}
	else if (cat == TENSOR_FLUID)
	{
		model->num_fluid_tensors++;
		printf("[FLUID] Adapter tensor: %s\n", model->tensors[idx].name);
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
