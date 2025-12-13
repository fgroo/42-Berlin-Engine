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
#  define NESTED_LR 0.00001f  // 1e-5: Much lower to preserve pre-trained features
# endif

# ifndef LEARNING_THRESHOLD
#  define LEARNING_THRESHOLD 2.0f  // Skip tokens where model isn't surprised
# endif

# ifndef GRADIENT_CLIP
#  define GRADIENT_CLIP 0.5f  // Per-element gradient clip
# endif

# ifndef GRADIENT_NORM_CLIP
#  define GRADIENT_NORM_CLIP 1.0f  // Global gradient norm clip
# endif

# ifndef MAX_GEN_LEN
#  define MAX_GEN_LEN 64
# endif

# ifndef REPETITION_PENALTY
#  define REPETITION_PENALTY 1.15f
# endif

# ifndef SPARSE_K
#  define SPARSE_K 64
# endif

// Layer freezing: Skip updating the first N layers during nested learning
// With 26 layers, FROZEN_LAYERS=24 means only top 2 layers train (safe!)
# ifndef FROZEN_LAYERS
#  define FROZEN_LAYERS 24
# endif

#endif
