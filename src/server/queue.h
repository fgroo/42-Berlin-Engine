/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   queue.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 23:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 23:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef QUEUE_H
# define QUEUE_H

# include <pthread.h>
# include <stdbool.h>

/*
** ============================================================================
** THREAD-SAFE JOB QUEUE (Phase 8)
** ============================================================================
** Producer-Consumer pattern for async inference.
** 
** Main thread (Producer): Accepts connections, parses requests, pushes jobs
** Worker thread (Consumer): Pops jobs, runs inference, streams responses
**
** Uses POSIX mutex + condition variables for thread synchronization.
** ============================================================================
*/

/* Maximum pending requests before blocking producers */
# define QUEUE_DEFAULT_CAPACITY 16

/*
** Job structure - encapsulates a single inference request
*/
typedef struct s_job
{
	int		client_fd;        /* Socket to write response to */
	char	*prompt;          /* User message to process */
	int		stream;           /* 1 = SSE streaming, 0 = wait for full response */
	int		max_tokens;       /* Maximum tokens to generate */
	float	temperature;      /* Sampling temperature */
	/* Phase 9: DeepSeek R1 thinking support */
	int		enable_thinking;  /* 1 = stream reasoning_content first */
	int		thinking_budget;  /* Max tokens for thinking phase */
}	t_job;

/*
** Thread-safe ring buffer queue
*/
typedef struct s_job_queue
{
	t_job			*buffer;        /* Ring buffer array */
	int				capacity;       /* Max jobs in queue */
	int				head;           /* Next write position */
	int				tail;           /* Next read position */
	int				count;          /* Current jobs in queue */
	pthread_mutex_t	lock;           /* Mutex for thread safety */
	pthread_cond_t	not_empty;      /* Signal: queue has jobs */
	pthread_cond_t	not_full;       /* Signal: queue has space */
	int				shutdown;       /* 1 = shutdown requested */
}	t_job_queue;

/*
** Queue lifecycle
*/
t_job_queue	*queue_init(int capacity);
void		queue_free(t_job_queue *q);

/*
** Producer API (main thread)
** Blocks if queue is full
*/
void		queue_push(t_job_queue *q, t_job job);

/*
** Consumer API (worker thread)
** Blocks if queue is empty
** Returns job with client_fd = -1 on shutdown
*/
t_job		queue_pop(t_job_queue *q);

/*
** Signal all waiting threads to wake up and exit
*/
void		queue_shutdown(t_job_queue *q);

#endif
