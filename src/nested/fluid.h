/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FLUID_H
# define FLUID_H

# include "tensor/tensor.h"

typedef struct s_fluid_param
{
	t_tensor	*weight;
	t_tensor	*grad;
}	t_fluid_param;

typedef struct s_fluid_param_momentum
{
	t_tensor	*weight;
	t_tensor	*grad;
	t_tensor	*velocity;
}	t_fluid_param_momentum;

void	backward_linear(t_fluid_param *param, const t_tensor *x,
			const t_tensor *grad_output);
void	optimizer_sgd(t_fluid_param *param, float lr);
void	optimizer_sgd_momentum(t_fluid_param_momentum *param,
			float lr, float momentum);
void	zero_grad(t_fluid_param *param);

#endif
