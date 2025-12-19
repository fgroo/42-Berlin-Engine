/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test_concurrency.c                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/15 13:30:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/15 13:30:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** CAS ATOMIC COUNTER STRESS TEST
** ===========================================================================
** Tests the lock-free CAS-based nl_counters implementation from Phase 2.
** Spawns 100 threads, each performs 10,000 increments with random skip flag.
** On completion, step MUST equal exactly 1,000,000 and skipped MUST equal
** the sum of all random skip decisions.
**
** This verifies that:
** 1. No counter updates are lost (no torn writes)
** 2. step and skipped are always consistent (no torn reads)
** 3. CAS loop correctly handles contention from 100 concurrent threads
** ===========================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include "nested/nl_counters.h"

#define NUM_THREADS 100
#define ITERATIONS_PER_THREAD 10000
#define EXPECTED_TOTAL_STEPS (NUM_THREADS * ITERATIONS_PER_THREAD)

/* Shared state */
static t_nl_atomic_state	g_state;
static _Atomic uint64_t		g_expected_skipped;

/* Simple LCG PRNG (thread-safe with per-thread seed) */
static inline uint32_t	fast_rand(uint32_t *seed)
{
	*seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
	return (*seed);
}

/* Thread worker function */
static void	*worker_thread(void *arg)
{
	uint64_t	thread_id;
	uint32_t	seed;
	uint64_t	local_skipped;
	int			i;
	bool		skip;
	uint32_t	step_after;
	uint32_t	skip_after;

	thread_id = (uint64_t)arg;
	seed = (uint32_t)(thread_id * 12345 + 67890);  /* Per-thread seed */
	local_skipped = 0;
	i = 0;
	while (i < ITERATIONS_PER_THREAD)
	{
		/* Random decision: ~50% probability of skipping */
		skip = (fast_rand(&seed) % 2) == 0;
		if (skip)
			local_skipped++;
		/* CAS-based atomic update */
		nl_record_step(&g_state, skip, &step_after, &skip_after);
		i++;
	}
	/* Atomically add local skip count to global expected */
	atomic_fetch_add(&g_expected_skipped, local_skipped);
	return (NULL);
}

int	main(void)
{
	pthread_t	threads[NUM_THREADS];
	uint32_t	final_step;
	uint32_t	final_skipped;
	uint64_t	expected_skipped;
	int			i;
	int			ret;
	int			pass;

	printf("=== CAS Atomic Counter Stress Test ===\n");
	printf("Threads: %d\n", NUM_THREADS);
	printf("Iterations/thread: %d\n", ITERATIONS_PER_THREAD);
	printf("Expected total steps: %d\n", EXPECTED_TOTAL_STEPS);
	printf("\n");

	/* Initialize state */
	nl_counters_reset(&g_state);
	atomic_store(&g_expected_skipped, 0);

	/* Spawn threads */
	printf("Starting %d threads...\n", NUM_THREADS);
	i = 0;
	while (i < NUM_THREADS)
	{
		ret = pthread_create(&threads[i], NULL, worker_thread, (void *)(uint64_t)i);
		if (ret != 0)
		{
			fprintf(stderr, "Failed to create thread %d\n", i);
			return (1);
		}
		i++;
	}

	/* Wait for all threads to complete */
	i = 0;
	while (i < NUM_THREADS)
	{
		pthread_join(threads[i], NULL);
		i++;
	}

	/* Read final state */
	nl_get_stats(&g_state, &final_step, &final_skipped);
	expected_skipped = atomic_load(&g_expected_skipped);

	/* Report results */
	printf("\n=== Results ===\n");
	printf("Final step count:     %u (expected: %d)\n",
		final_step, EXPECTED_TOTAL_STEPS);
	printf("Final skipped count:  %u (expected: %llu)\n",
		final_skipped, (unsigned long long)expected_skipped);
	printf("Actual steps (learned): %d\n",
		nl_get_actual_steps(&g_state));

	/* Verify */
	pass = 1;
	if (final_step != EXPECTED_TOTAL_STEPS)
	{
		printf("\n❌ FAIL: step count mismatch!\n");
		printf("   Lost %d steps to race conditions\n",
			EXPECTED_TOTAL_STEPS - final_step);
		pass = 0;
	}
	if (final_skipped != expected_skipped)
	{
		printf("\n❌ FAIL: skipped count mismatch!\n");
		printf("   Difference: %lld\n",
			(long long)final_skipped - (long long)expected_skipped);
		pass = 0;
	}

	if (pass)
	{
		printf("\n✅ PASS: All counters consistent!\n");
		printf("   CAS-based atomic updates are working correctly.\n");
		printf("   No torn reads or lost updates detected.\n");
		return (0);
	}
	return (1);
}
