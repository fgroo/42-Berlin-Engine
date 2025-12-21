/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   worker.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 23:30:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 23:30:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef WORKER_H
# define WORKER_H

# include "queue.h"
# include "inference/inference.h"
# include "tokenizer/tokenizer.h"
# include <pthread.h>

/*
** ============================================================================
** INFERENCE WORKER (Phase 8)
** ============================================================================
** Dedicated thread for running inference, decoupled from network I/O.
** Streams tokens via Server-Sent Events (SSE) for responsive UX.
**
** OpenAI SSE Format:
**   data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}
**   
**   data: [DONE]
** ============================================================================
*/

/*
** Worker context - shared between main thread and worker
*/
typedef struct s_worker_ctx
{
	t_job_queue		*queue;      /* Job queue to consume from */
	t_transformer	*engine;     /* Inference engine */
	t_tokenizer		*tokenizer;  /* Tokenizer */
	int				running;     /* 1 = running, 0 = should exit */
}	t_worker_ctx;

/*
** Start worker thread
** Returns 0 on success, -1 on error
*/
int		worker_start(t_worker_ctx *ctx, pthread_t *thread);

/*
** Worker thread entry point
*/
void	*worker_routine(void *arg);

/*
** Stop worker thread (signals shutdown and joins)
*/
void	worker_stop(t_worker_ctx *ctx, pthread_t thread);

#endif
