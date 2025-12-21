/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   queue.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 23:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 23:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "queue.h"
#include <stdlib.h>
#include <string.h>

/*
** Initialize a new job queue with given capacity
*/
t_job_queue	*queue_init(int capacity)
{
	t_job_queue	*q;

	q = (t_job_queue *)calloc(1, sizeof(t_job_queue));
	if (!q)
		return (NULL);
	q->buffer = (t_job *)calloc(capacity, sizeof(t_job));
	if (!q->buffer)
	{
		free(q);
		return (NULL);
	}
	q->capacity = capacity;
	q->head = 0;
	q->tail = 0;
	q->count = 0;
	q->shutdown = 0;
	pthread_mutex_init(&q->lock, NULL);
	pthread_cond_init(&q->not_empty, NULL);
	pthread_cond_init(&q->not_full, NULL);
	return (q);
}

/*
** Free queue resources
*/
void	queue_free(t_job_queue *q)
{
	int	i;

	if (!q)
		return ;
	/* Free any pending job prompts */
	i = 0;
	while (i < q->capacity)
	{
		if (q->buffer[i].prompt)
			free(q->buffer[i].prompt);
		i++;
	}
	free(q->buffer);
	pthread_mutex_destroy(&q->lock);
	pthread_cond_destroy(&q->not_empty);
	pthread_cond_destroy(&q->not_full);
	free(q);
}

/*
** Push a job to the queue (producer)
** Blocks if queue is full until space is available
*/
void	queue_push(t_job_queue *q, t_job job)
{
	pthread_mutex_lock(&q->lock);
	/* Wait while queue is full (and not shutting down) */
	while (q->count >= q->capacity && !q->shutdown)
		pthread_cond_wait(&q->not_full, &q->lock);
	if (q->shutdown)
	{
		pthread_mutex_unlock(&q->lock);
		return ;
	}
	/* Insert job at head */
	q->buffer[q->head] = job;
	q->head = (q->head + 1) % q->capacity;
	q->count++;
	/* Signal waiting consumers */
	pthread_cond_signal(&q->not_empty);
	pthread_mutex_unlock(&q->lock);
}

/*
** Pop a job from the queue (consumer)
** Blocks if queue is empty until job is available
** Returns job with client_fd = -1 on shutdown
*/
t_job	queue_pop(t_job_queue *q)
{
	t_job	job;

	memset(&job, 0, sizeof(job));
	job.client_fd = -1;
	pthread_mutex_lock(&q->lock);
	/* Wait while queue is empty (and not shutting down) */
	while (q->count == 0 && !q->shutdown)
		pthread_cond_wait(&q->not_empty, &q->lock);
	if (q->shutdown && q->count == 0)
	{
		pthread_mutex_unlock(&q->lock);
		return (job);  /* Return sentinel job */
	}
	/* Remove job from tail */
	job = q->buffer[q->tail];
	q->buffer[q->tail].prompt = NULL;  /* Prevent double-free */
	q->tail = (q->tail + 1) % q->capacity;
	q->count--;
	/* Signal waiting producers */
	pthread_cond_signal(&q->not_full);
	pthread_mutex_unlock(&q->lock);
	return (job);
}

/*
** Signal shutdown to all waiting threads
*/
void	queue_shutdown(t_job_queue *q)
{
	pthread_mutex_lock(&q->lock);
	q->shutdown = 1;
	/* Wake up all waiting threads */
	pthread_cond_broadcast(&q->not_empty);
	pthread_cond_broadcast(&q->not_full);
	pthread_mutex_unlock(&q->lock);
}
