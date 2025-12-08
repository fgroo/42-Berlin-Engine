/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test_tokenizer.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: antigravity <antigravity@student.42.fr>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 14:45:00 by antigravity       #+#    #+#             */
/*   Updated: 2025/12/05 14:45:00 by antigravity      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int	main(int argc, char **argv)
{
	t_tokenizer	t;
	int			*tokens;
	int			num_tokens;
	int			i;
	const char	*json_path = "Ministral-Stuff/tokenizer.json";

	if (argc > 1)
		json_path = argv[1];

	printf("Loading tokenizer from %s...\n", json_path);
	if (tokenizer_init(&t, json_path) != 0)
	{
		fprintf(stderr, "Failed to load tokenizer\n");
		return (1);
	}
	printf("Tokenizer loaded. Vocab size: %d\n", t.vocab_size);

	const char *text = "Hello World";
	printf("Encoding: '%s'\n", text);
	num_tokens = tokenizer_encode(&t, text, &tokens);
	
	printf("Tokens: ");
	for (i = 0; i < num_tokens; i++)
	{
		printf("%d ", tokens[i]);
	}
	printf("\n");

	printf("Decoding: ");
	for (i = 0; i < num_tokens; i++)
	{
		printf("%s", tokenizer_decode(&t, tokens[i]));
	}
	printf("\n");

	free(tokens);
	tokenizer_free(&t);
	return (0);
}
