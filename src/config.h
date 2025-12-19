/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   config.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/07 00:00:00 by fgroo            #+#    #+#             */
/*   Updated: 2025/12/07 00:00:00 by fgroo           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef CONFIG_H
# define CONFIG_H

/* Debug mode - comment out for release builds */
/* #define DEBUG_MODE */

# ifdef DEBUG_MODE
#  define LOG_DEBUG(...) printf(__VA_ARGS__)
# else
#  define LOG_DEBUG(...) ((void)0)
# endif

# ifndef TEMPERATURE
#  define TEMPERATURE 0.1f  // Cold & Precise for logic tasks
# endif

# ifndef TOP_P
#  define TOP_P 0.1f  // Tight nucleus - only best tokens
# endif

# ifndef NESTED_LR
#  define NESTED_LR 0.00001f  // 1e-5: Much lower to preserve pre-trained features
# endif

# ifndef LEARNING_THRESHOLD
#  define LEARNING_THRESHOLD 0.01f  // Learn everything
# endif

# ifndef HIGH_LOSS_THRESHOLD
#  define HIGH_LOSS_THRESHOLD 20.0f  // Allow learning from high-loss tokens (gradient is strongest there)
# endif

# ifndef GRADIENT_CLIP
#  define GRADIENT_CLIP 0.5f  // Per-element gradient clip
# endif

# ifndef GRADIENT_NORM_CLIP
#  define GRADIENT_NORM_CLIP 1.0f  // Global gradient norm clip
# endif

# ifndef ADAPTER_SCALE
#  define ADAPTER_SCALE 1.0f  // Subtle contribution, allows context_bias to lead
# endif

# ifndef MAX_GEN_LEN
#  define MAX_GEN_LEN 64
# endif

# ifndef REPETITION_PENALTY
#  define REPETITION_PENALTY 1.2f  // Against "not a girl, not a girl" loops
# endif

# ifndef SPARSE_K
#  define SPARSE_K 64
# endif

// Number of blocks to attend to in sparse paged mode
// At 64 blocks, we get ~25% coverage which improves needle retrieval
# ifndef SPARSE_BLOCKS_K
#  define SPARSE_BLOCKS_K 64
# endif

// Sliding window for hybrid attention - always attend to last W tokens
// This ensures local reasoning (arithmetic, syntax) works even if LSH misses
# ifndef ATTN_WINDOW_SIZE
#  define ATTN_WINDOW_SIZE 128
# endif

// Layer freezing: Skip updating the first N layers during nested learning
// With 26 layers, FROZEN_LAYERS=24 means only top 2 layers train (safe!)
# ifndef FROZEN_LAYERS
#  define FROZEN_LAYERS 24
# endif

// Max steps per turn to prevent overfitting and mode collapse
# ifndef NL_MAX_STEPS
#  define NL_MAX_STEPS 10
# endif

// ========== LSH DIAGNOSTICS (Phase 8) ==========
// Enable runtime LSH recall validation (expensive - samples every N queries)
# ifndef DEBUG_LSH
#  define DEBUG_LSH 0  /* Production: 0, Debug: 1 */
# endif

// How often to validate LSH recall (every N sparse queries)
// Set to 1 for calibration, 100 for production
# ifndef LSH_VALIDATION_INTERVAL
#  define LSH_VALIDATION_INTERVAL 100  /* PRODUCTION MODE */
# endif

// Adaptive K probability mass threshold (0.95 = 95% of attention mass)
# ifndef ADAPTIVE_K_THRESHOLD
#  define ADAPTIVE_K_THRESHOLD 0.95f
# endif

// ========== CENTRALIZED MAGIC NUMBERS ==========
// Previously hardcoded throughout codebase, now configurable

# ifndef MAX_DIM
#  define MAX_DIM 8192              /* Maximum model dimension for stack alloc */
# endif

# ifndef CONTEXT_BIAS_SIZE
#  define CONTEXT_BIAS_SIZE 65536   /* Bigram context cache size */
# endif

# ifndef MAX_CANDIDATE_BLOCKS
#  define MAX_CANDIDATE_BLOCKS 64   /* LSH candidate block limit */
# endif

# ifndef LINEAR_PROBE_LIMIT
#  define LINEAR_PROBE_LIMIT 16     /* Hash table probe limit */
# endif

#endif
