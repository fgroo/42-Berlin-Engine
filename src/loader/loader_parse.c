/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   loader_parse.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "loader.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

int		extract_name(t_model *model, const char *qs, const char *qe);
void	parse_shape(t_model *model, const char *obj_start);

static size_t	parse_offset(const char *obj_start)
{
	const char	*off;
	const char	*arr;

	off = strstr(obj_start, "\"data_offsets\"");
	if (!off)
		return (0);
	arr = strchr(off, '[');
	if (!arr)
		return (0);
	return (atoll(arr + 1));
}

static void	setup_tensor(t_model *model, const char *obj_start)
{
	size_t		start_off;
	uint8_t		*base;
	t_bf16		*data;
	int			idx;

	idx = model->num_tensors;
	start_off = parse_offset(obj_start);
	base = (uint8_t *)model->mapped_addr + 8 + model->header_size;
	data = (t_bf16 *)(base + start_off);
	model->tensors[idx].tensor = tensor_view(data,
			model->tensors[idx].tensor.shape, model->tensors[idx].tensor.ndim);
	model->tensors[idx].tensor.dtype = DTYPE_BF16;
}

static const char	*skip_object(const char *start)
{
	const char	*end;

	end = strchr(start, '}');
	if (!end)
		return (NULL);
	end++;
	while (*end && (*end == ' ' || *end == ',' || *end == '\n'))
		end++;
	return (end);
}

static int	parse_one_tensor(t_model *model, const char **pp, const char *end)
{
	const char	*qs;
	const char	*qe;
	const char	*obj;

	qs = strchr(*pp, '"');
	if (!qs || qs >= end)
		return (0);
	qe = strchr(qs + 1, '"');
	if (!qe)
		return (0);
	if (!extract_name(model, qs, qe))
	{
		obj = strchr(qe, '{');
		*pp = skip_object(obj);
		return (*pp != NULL);
	}
	obj = strchr(qe, '{');
	if (!obj)
		return (0);
	parse_shape(model, obj);
	setup_tensor(model, obj);
	model->num_tensors++;
	*pp = skip_object(obj);
	return (*pp != NULL);
}

void	parse_header(t_model *model, const char *header, size_t size)
{
	const char	*p;
	const char	*end;

	model->num_tensors = 0;
	p = header;
	end = header + size;
	while (*p && *p != '{')
		p++;
	if (!*p)
		return ;
	p++;
	while (p < end && model->num_tensors < MAX_TENSORS)
	{
		if (!parse_one_tensor(model, &p, end))
			break ;
	}
}
