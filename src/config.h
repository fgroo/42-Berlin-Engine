/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   config.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: marvin <marvin@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/07 00:00:00 by marvin            #+#    #+#             */
/*   Updated: 2025/12/07 00:00:00 by marvin           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef CONFIG_H
# define CONFIG_H

# ifndef TEMPERATURE
#  define TEMPERATURE 0.7f
# endif

# ifndef TOP_P
#  define TOP_P 0.9f
# endif

# ifndef NESTED_LR
#  define NESTED_LR 0.0005f
# endif

# ifndef LEARNING_THRESHOLD
#  define LEARNING_THRESHOLD 2.0f
# endif

# ifndef GRADIENT_CLIP
#  define GRADIENT_CLIP 1.0f
# endif

# ifndef MAX_GEN_LEN
#  define MAX_GEN_LEN 48
# endif

# ifndef REPETITION_PENALTY
#  define REPETITION_PENALTY 1.15f
# endif

# ifndef SPARSE_K
#  define SPARSE_K 64
# endif

// Layer freezing: Skip updating the first N layers during nested learning
// Upper layers (closer to output) typically contain more task-specific knowledge
# ifndef FROZEN_LAYERS
#  define FROZEN_LAYERS 16
# endif

#endif
