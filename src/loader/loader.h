/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   loader.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LOADER_H
# define LOADER_H

# include "tensor/tensor.h"
# include <stddef.h>

# define MAX_TENSORS 512
# define MAX_NAME_LEN 128

typedef struct s_named_tensor
{
	char		name[MAX_NAME_LEN];
	t_tensor	tensor;
}	t_named_tensor;

typedef struct s_model
{
	void			*mapped_addr;
	size_t			file_size;
	size_t			header_size;
	t_named_tensor	tensors[MAX_TENSORS];
	int				num_tensors;
}	t_model;

int			load_model(t_model *model, const char *path);
t_tensor	*get_tensor(t_model *model, const char *name);
void		free_model(t_model *model);

#endif
