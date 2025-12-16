/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tensor.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TENSOR_H
# define TENSOR_H

# include <stdint.h>
# include <stddef.h>

typedef uint16_t	t_bf16;

typedef enum e_dtype
{
	DTYPE_F32,
	DTYPE_BF16
}	t_dtype;

typedef struct s_tensor
{
	void	*data;
	int		shape[4];
	int		stride[4];
	int		ndim;
	size_t	size;
	t_dtype	dtype;
}	t_tensor;

t_tensor	tensor_view(void *data, int shape[], int ndim);
t_bf16		float_to_bf16(float f);
float		bf16_to_float(t_bf16 b);

#endif
