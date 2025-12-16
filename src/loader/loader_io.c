/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   loader_io.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "loader.h"
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>

void	parse_header(t_model *model, const char *header, size_t size);

static int	open_and_stat(const char *path, int *fd, size_t *file_size)
{
	struct stat	sb;

	*fd = open(path, O_RDONLY);
	if (*fd == -1)
	{
		perror("open");
		return (-1);
	}
	if (fstat(*fd, &sb) == -1)
	{
		perror("fstat");
		close(*fd);
		return (-1);
	}
	*file_size = sb.st_size;
	return (0);
}

int	load_model(t_model *model, const char *path)
{
	int			fd;
	uint64_t	header_size_le;
	const char	*header_start;

	/* Initialize counters */
	model->num_vision_tensors = 0;
	model->num_fluid_tensors = 0;
	
	if (open_and_stat(path, &fd, &model->file_size) < 0)
		return (-1);
	model->mapped_addr = mmap(NULL, model->file_size, PROT_READ, MAP_PRIVATE,
			fd, 0);
	if (model->mapped_addr == MAP_FAILED)
	{
		perror("mmap");
		close(fd);
		return (-1);
	}
	close(fd);
	header_size_le = *(uint64_t *)model->mapped_addr;
	model->header_size = header_size_le;
	header_start = (const char *)model->mapped_addr + 8;
	parse_header(model, header_start, model->header_size);
	return (0);
}

t_tensor	*get_tensor(t_model *model, const char *name)
{
	int	i;

	i = 0;
	while (i < model->num_tensors)
	{
		if (strcmp(model->tensors[i].name, name) == 0)
			return (&model->tensors[i].tensor);
		i++;
	}
	printf("WARN: Tensor not found: %s\n", name);
	return (NULL);
}

/*
** Get the nth tensor of a specific category
** Useful for iterating over all VISION or FLUID tensors
*/
t_tensor	*get_tensor_by_category(t_model *model, t_tensor_category cat,
				int index)
{
	int	i;
	int	count;

	i = 0;
	count = 0;
	while (i < model->num_tensors)
	{
		if (model->tensors[i].category == cat)
		{
			if (count == index)
				return (&model->tensors[i].tensor);
			count++;
		}
		i++;
	}
	return (NULL);
}

void	free_model(t_model *model)
{
	if (model->mapped_addr)
	{
		munmap(model->mapped_addr, model->file_size);
		model->mapped_addr = NULL;
	}
}
