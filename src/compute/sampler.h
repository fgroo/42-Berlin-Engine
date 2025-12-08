/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   sampler.h                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/05 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef SAMPLER_H
# define SAMPLER_H

# include "tensor/tensor.h"
# include "memory/arena.h"

int	sample_argmax(const t_tensor *logits);
int	sample_temperature(const t_tensor *logits, float temperature,
		t_arena *scratch);
int	sample_top_p(const t_tensor *logits, float temperature,
		float top_p, t_arena *scratch);

#endif
