/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   loader_vision.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/12 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/12 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "loader.h"
#include "../inference/inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
** Extract ViT layer index from tensor name
** e.g., "vision_model.encoder.layers.12.attn.weight" -> 12
** Returns -1 if not a layer tensor
*/
static int	extract_vit_layer_idx(const char *name)
{
	const char	*layers;
	int			idx;

	layers = strstr(name, "layers.");
	if (!layers)
		return (-1);
	idx = atoi(layers + 7);
	if (idx < 0 || idx > 127)
		return (-1);
	return (idx);
}

/*
** Count vision ViT layers by scanning tensor names for max layer index
*/
static int	count_vit_layers(t_model *model)
{
	int		max_idx;
	int		i;
	int		layer_idx;

	max_idx = -1;
	i = 0;
	while (i < model->num_tensors)
	{
		if (model->tensors[i].category == TENSOR_VISION)
		{
			layer_idx = extract_vit_layer_idx(model->tensors[i].name);
			if (layer_idx > max_idx)
				max_idx = layer_idx;
		}
		i++;
	}
	return (max_idx + 1);
}

/*
** Activate vision tower - maps pointers to mmap region
** Calling this causes OS to page vision weights into RAM
** MUST allocate vit_layers array before filling pointers
*/
int	activate_vision_tower(t_transformer *t)
{
	t_vision_tower	*vt;
	int				n_layers;
	int				i;
	int				layer_idx;
	const char		*name;
	t_tensor		*tensor;

	if (t->vision && t->vision->enabled)
	{
		printf("[VISION] Already active\n");
		return (0);
	}
	/* Count and allocate */
	n_layers = count_vit_layers(&t->model);
	if (n_layers == 0)
	{
		printf("[VISION] No ViT layers found\n");
		return (-1);
	}
	/* Allocate vision tower struct if not exists */
	if (!t->vision)
	{
		t->vision = calloc(1, sizeof(t_vision_tower));
		if (!t->vision)
			return (-1);
	}
	vt = t->vision;
	/* Pre-allocate pointer array (CRITICAL: avoids NULL deref) */
	if (!vt->vit_layers)
	{
		vt->vit_layers = calloc(n_layers, sizeof(t_tensor *));
		vt->num_vit_layers = n_layers;
	}
	/* Map tensor pointers - this triggers OS paging! */
	i = 0;
	while (i < t->model.num_tensors)
	{
		if (t->model.tensors[i].category != TENSOR_VISION)
		{
			i++;
			continue ;
		}
		name = t->model.tensors[i].name;
		tensor = &t->model.tensors[i].tensor;
		if (strstr(name, "patch_embed") || strstr(name, "patch_embedding"))
			vt->patch_embed = tensor;
		else if (strstr(name, "pos_embed") || strstr(name, "position"))
			vt->pos_embed = tensor;
		else if (strstr(name, "projector") || strstr(name, "multi_modal"))
			vt->projector = tensor;
		else
		{
			layer_idx = extract_vit_layer_idx(name);
			if (layer_idx >= 0 && layer_idx < vt->num_vit_layers)
				vt->vit_layers[layer_idx] = tensor;
		}
		i++;
	}
	vt->enabled = 1;
	printf("[VISION] Tower activated - %d layers, weights now paged\n",
		vt->num_vit_layers);
	return (0);
}

/*
** Deactivate vision - sets enabled=0
** Weights stay in mmap but won't be accessed
** OS may evict pages over time (LRU)
*/
void	deactivate_vision_tower(t_transformer *t)
{
	if (!t->vision)
		return ;
	t->vision->enabled = 0;
	printf("[VISION] Tower deactivated - weights remain lazy\n");
}

/*
** Free vision tower resources
*/
void	free_vision_tower(t_vision_tower *vt)
{
	if (!vt)
		return ;
	if (vt->vit_layers)
		free(vt->vit_layers);
	free(vt);
}
