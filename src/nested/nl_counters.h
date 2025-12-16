/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   nl_counters.h                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 13:30:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/15 13:30:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NL_COUNTERS_H
# define NL_COUNTERS_H

/*
** ===========================================================================
** LOCK-FREE ATOMIC COUNTERS FOR NESTED LEARNING (Phase 2)
** ===========================================================================
** Problem: Using separate atomic_int for step/skipped creates torn reads.
** Thread A increments step, context switch, Thread B increments both,
** Thread A wakes and increments skipped -> inconsistent ratio in logs.
**
** Solution: Pack both 32-bit counters into a single 64-bit atomic.
** CAS (Compare-And-Swap) ensures the update is a single CPU instruction.
** ===========================================================================
*/

# include <stdatomic.h>
# include <stdint.h>
# include <stdbool.h>

/*
** Bit-packed counters: Low 32 bits = step, high 32 bits = skipped
** 4 billion steps per session is sufficient.
*/
typedef union u_nl_counters
{
	uint64_t	combined;
	struct
	{
		uint32_t	step;     /* Low 32 bits */
		uint32_t	skipped;  /* High 32 bits */
	};
}	t_nl_counters;

/*
** Atomic state for nested learning.
** actual_steps is separate - rarely contended, different semantic.
*/
typedef struct s_nl_atomic_state
{
	_Atomic uint64_t	counters;      /* Packed step + skipped */
	_Atomic int32_t		actual_steps;  /* Per-turn actual learning steps */
}	t_nl_atomic_state;

/*
** ==========================================================================
** CAS-BASED UPDATE FUNCTIONS
** ==========================================================================
*/

/*
** Reset all counters to zero (start of turn)
*/
static inline void	nl_counters_reset(t_nl_atomic_state *s)
{
	atomic_store_explicit(&s->counters, 0ULL, memory_order_release);
	atomic_store_explicit(&s->actual_steps, 0, memory_order_release);
}

/*
** Record a step (with optional skip) using lock-free CAS loop.
** Returns the state AFTER the update (useful for logging).
**
** @param s: Atomic state
** @param was_skipped: true if this step was skipped (not learned)
** @param out_step: Output - step count after update
** @param out_skipped: Output - skipped count after update
*/
static inline void	nl_record_step(t_nl_atomic_state *s, bool was_skipped,
						uint32_t *out_step, uint32_t *out_skipped)
{
	t_nl_counters	old_val;
	t_nl_counters	new_val;

	/* Load current value (relaxed ok for first load) */
	old_val.combined = atomic_load_explicit(&s->counters,
			memory_order_relaxed);
	do
	{
		/* Compute new state locally */
		new_val = old_val;
		new_val.step++;
		if (was_skipped)
			new_val.skipped++;
		/* CAS: Try to write new_val, but only if memory still has old_val.
		** On failure, old_val is updated with current memory value. */
	}
	while (!atomic_compare_exchange_weak_explicit(
			&s->counters,
			&old_val.combined,
			new_val.combined,
			memory_order_release,   /* Success ordering */
			memory_order_relaxed)); /* Failure ordering */
	/* Output the state we just wrote (guaranteed consistent) */
	if (out_step)
		*out_step = new_val.step;
	if (out_skipped)
		*out_skipped = new_val.skipped;
}

/*
** Increment actual_steps counter. Returns value BEFORE increment.
*/
static inline int32_t	nl_inc_actual_steps(t_nl_atomic_state *s)
{
	return (atomic_fetch_add_explicit(&s->actual_steps, 1,
			memory_order_acq_rel));
}

/*
** Get current actual_steps count.
*/
static inline int32_t	nl_get_actual_steps(t_nl_atomic_state *s)
{
	return (atomic_load_explicit(&s->actual_steps, memory_order_acquire));
}

/*
** Get current counters (atomic snapshot - no torn reads).
*/
static inline void	nl_get_stats(t_nl_atomic_state *s,
						uint32_t *step, uint32_t *skipped)
{
	t_nl_counters	current;

	current.combined = atomic_load_explicit(&s->counters,
			memory_order_acquire);
	if (step)
		*step = current.step;
	if (skipped)
		*skipped = current.skipped;
}

#endif
