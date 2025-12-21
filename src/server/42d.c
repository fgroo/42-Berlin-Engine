/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   42d.c (42-Berlin-Engine Daemon)                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 21:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 21:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ============================================================================
** 42D - HTTP DAEMON FOR 42-BERLIN-ENGINE
** ============================================================================
** OpenAI-compatible inference server supporting:
**   - POST /v1/chat/completions
**   - GET  /v1/models
**   - GET  /health
**
** Usage:
**   ./42d -m model.safetensors -t tokenizer.json [-p 8080]
**
** Example:
**   curl http://localhost:8080/v1/chat/completions \
**     -H "Content-Type: application/json" \
**     -d '{"messages":[{"role":"user","content":"Hello!"}]}'
** ============================================================================
*/

#include "server/server.h"
#include "inference/inference.h"
#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

/* Global for signal handler */
static t_server	*g_server = NULL;

static void	signal_handler(int sig)
{
	(void)sig;
	printf("\n[42d] Received shutdown signal...\n");
	if (g_server)
		g_server->running = 0;
}

static void	print_usage(const char *prog)
{
	printf("Usage: %s [OPTIONS]\n\n", prog);
	printf("Options:\n");
	printf("  -m, --model PATH       Path to safetensors model (required)\n");
	printf("  -t, --tokenizer PATH   Path to tokenizer.json (required)\n");
	printf("  -c, --config PATH      Path to config.json\n");
	printf("  -p, --port PORT        HTTP port (default: 8080)\n");
	printf("  -h, --help             Show this help\n");
	printf("\n");
	printf("Example:\n");
	printf("  %s -m model.safetensors -t tokenizer.json -p 8080\n", prog);
	printf("\n");
}

int	main(int argc, char **argv)
{
	char			*model_path;
	char			*tokenizer_path;
	char			*config_path;
	int				port;
	int				i;
	t_transformer	transformer;
	t_tokenizer		tokenizer;
	t_server		server;

	/* Parse arguments */
	model_path = NULL;
	tokenizer_path = NULL;
	config_path = NULL;
	port = SERVER_DEFAULT_PORT;
	i = 1;
	while (i < argc)
	{
		if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0)
			&& i + 1 < argc)
		{
			model_path = argv[++i];
		}
		else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokenizer") == 0)
			&& i + 1 < argc)
		{
			tokenizer_path = argv[++i];
		}
		else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0)
			&& i + 1 < argc)
		{
			config_path = argv[++i];
		}
		else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0)
			&& i + 1 < argc)
		{
			port = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			print_usage(argv[0]);
			return (0);
		}
		else
		{
			fprintf(stderr, "Unknown option: %s\n", argv[i]);
			print_usage(argv[0]);
			return (1);
		}
		i++;
	}

	/* Validate required args */
	if (!model_path || !tokenizer_path)
	{
		fprintf(stderr, "Error: -m (model) and -t (tokenizer) are required\n\n");
		print_usage(argv[0]);
		return (1);
	}

	/* Initialize tokenizer */
	printf("[42d] Loading tokenizer: %s\n", tokenizer_path);
	if (tokenizer_init(&tokenizer, tokenizer_path) != 0)
	{
		fprintf(stderr, "Error: Failed to load tokenizer\n");
		return (1);
	}
	printf("[42d] Tokenizer loaded. Vocab size: %d\n", tokenizer.vocab_size);

	/* Initialize transformer */
	printf("[42d] Loading model: %s\n", model_path);
	if (transformer_init(&transformer, model_path, config_path) != 0)
	{
		fprintf(stderr, "Error: Failed to load model\n");
		tokenizer_free(&tokenizer);
		return (1);
	}
	printf("[42d] Model loaded. Layers: %d, Dim: %d\n",
		transformer.config.n_layers, transformer.config.dim);

	/* Set up signal handlers */
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	signal(SIGPIPE, SIG_IGN);  /* Ignore broken pipe for network robustness */

	/* Initialize and run server */
	g_server = &server;
	if (server_init(&server, port, &transformer, &tokenizer) != 0)
	{
		fprintf(stderr, "Error: Failed to start server\n");
		transformer_free(&transformer);
		tokenizer_free(&tokenizer);
		return (1);
	}

	/* Run event loop */
	server_run(&server);

	/* Cleanup */
	server_shutdown(&server);
	transformer_free(&transformer);
	tokenizer_free(&tokenizer);
	printf("[42d] Goodbye.\n");
	return (0);
}
