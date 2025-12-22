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
#include "inference/speculate.h"
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
	printf("  --draft PATH           Path to draft model for MTP (optional)\n");
	printf("  --draft-tokenizer PATH Path to draft tokenizer (required with --draft)\n");
	printf("  -h, --help             Show this help\n");
	printf("\n");
	printf("Example:\n");
	printf("  %s -m model.safetensors -t tokenizer.json -p 8080\n", prog);
	printf("\n");
	printf("MTP (Speculative Decoding):\n");
	printf("  %s -m ministral.safetensors -t ministral-tok.json \\\n", prog);
	printf("      --draft gemma.safetensors --draft-tokenizer gemma-tok.json\n");
	printf("\n");
}

int	main(int argc, char **argv)
{
	char			*model_path;
	char			*tokenizer_path;
	char			*config_path;
	char			*draft_path;
	char			*draft_tok_path;
	char			*draft_config_path;
	int				port;
	int				i;
	t_transformer	transformer;
	t_transformer	draft_transformer;
	t_tokenizer		tokenizer;
	t_tokenizer		draft_tokenizer;
	t_mtp_engine	mtp_engine;
	t_server		server;
	int				has_draft;

	/* Parse arguments */
	model_path = NULL;
	tokenizer_path = NULL;
	config_path = NULL;
	draft_path = NULL;
	draft_tok_path = NULL;
	draft_config_path = NULL;
	port = SERVER_DEFAULT_PORT;
	has_draft = 0;
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
		else if (strcmp(argv[i], "--draft") == 0 && i + 1 < argc)
		{
			draft_path = argv[++i];
		}
		else if (strcmp(argv[i], "--draft-tokenizer") == 0 && i + 1 < argc)
		{
			draft_tok_path = argv[++i];
		}
		else if (strcmp(argv[i], "--draft-config") == 0 && i + 1 < argc)
		{
			draft_config_path = argv[++i];
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
	if (draft_path && !draft_tok_path)
	{
		fprintf(stderr, "Error: --draft requires --draft-tokenizer\n\n");
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

	/* Initialize optional draft model for MTP */
	if (draft_path)
	{
		printf("[42d] Loading draft tokenizer: %s\n", draft_tok_path);
		if (tokenizer_init(&draft_tokenizer, draft_tok_path) != 0)
		{
			fprintf(stderr, "Warning: Failed to load draft tokenizer, MTP disabled\n");
		}
		else
		{
			printf("[42d] Loading draft model: %s\n", draft_path);
			if (transformer_init(&draft_transformer, draft_path, draft_config_path) != 0)
			{
				fprintf(stderr, "Warning: Failed to load draft model, MTP disabled\n");
				tokenizer_free(&draft_tokenizer);
			}
			else
			{
				has_draft = 1;
				printf("[42d] Draft model loaded. Layers: %d, Dim: %d\n",
					draft_transformer.config.n_layers, draft_transformer.config.dim);
			}
		}
	}

	/* Initialize MTP engine */
	if (has_draft)
	{
		if (mtp_init(&mtp_engine, &transformer, &draft_transformer,
				&tokenizer, &draft_tokenizer) != 0)
		{
			fprintf(stderr, "Warning: MTP init failed, using standard mode\n");
			has_draft = 0;
		}
	}
	else
	{
		/* No draft model - init MTP in fallback mode */
		mtp_init(&mtp_engine, &transformer, NULL, &tokenizer, NULL);
	}

	/* Set up signal handlers */
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	signal(SIGPIPE, SIG_IGN);  /* Ignore broken pipe for network robustness */

	/* Initialize and run server */
	g_server = &server;
	server.mtp = (struct s_mtp_engine *)&mtp_engine;  /* Wire MTP for burst gen */
	/* Disabled for production:
	printf("[DEBUG] 42d.c: server.mtp = %p, is_speculative = %d\n",
		(void *)server.mtp, mtp_engine.is_speculative);
	*/
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
	mtp_stats(&mtp_engine);
	mtp_free(&mtp_engine);
	if (has_draft)
	{
		transformer_free(&draft_transformer);
		tokenizer_free(&draft_tokenizer);
	}
	transformer_free(&transformer);
	tokenizer_free(&tokenizer);
	printf("[42d] Goodbye.\n");
	return (0);
}
