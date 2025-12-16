/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   loader.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/12 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LOADER_H
# define LOADER_H

# include "tensor/tensor.h"
# include <stddef.h>

# define MAX_TENSORS 512
# define MAX_NAME_LEN 128

/*
** Tensor Memory Classification for Lazy Loading and Nested Learning
** ==================================================================
** FROZEN: Base model weights - mmap read-only, never touched by backprop
** FLUID:  Adapter weights (LoRA) - copied to Arena RAM for gradient updates
** VISION: Vision encoder weights - lazy loaded, excluded from sparse attention
*/
typedef enum e_tensor_category
{
	TENSOR_FROZEN,
	TENSOR_FLUID,
	TENSOR_VISION
}	t_tensor_category;

typedef struct s_named_tensor
{
	char				name[MAX_NAME_LEN];
	t_tensor			tensor;
	t_tensor_category	category;
}	t_named_tensor;

typedef struct s_model
{
	void			*mapped_addr;
	size_t			file_size;
	size_t			header_size;
	t_named_tensor	tensors[MAX_TENSORS];
	int				num_tensors;
	int				num_vision_tensors;
	int				num_fluid_tensors;
}	t_model;

int			load_model(t_model *model, const char *path);
t_tensor	*get_tensor(t_model *model, const char *name);
t_tensor	*get_tensor_by_category(t_model *model, t_tensor_category cat,
				int index);
void		free_model(t_model *model);

#endif
