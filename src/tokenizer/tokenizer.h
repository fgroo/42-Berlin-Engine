/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tokenizer.h                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 14:20:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/05 14:20:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TOKENIZER_H
# define TOKENIZER_H

# include <stddef.h>

typedef struct s_tokenizer
{
	char	**vocab;
	int		vocab_size;
	void	*priv;
	int		bos_id;
	int		eos_id;
	int		unk_id;
}	t_tokenizer;

// Initialize tokenizer from JSON file
int				tokenizer_init(t_tokenizer *t, const char *json_path);

// Encode text to token IDs. Returns number of tokens.
// Caller must free *tokens.
int				tokenizer_encode(t_tokenizer *t, const char *text,
					int **tokens);

// Decode token ID to string. Returns pointer to internal string (do not free).
const char	*tokenizer_decode(t_tokenizer *t, int token_id);

// MOPD: Lookup token string to get its ID. Returns unk_id if not found.
// Used for teacher distillation where we receive token strings instead of IDs.
int			tokenizer_lookup_id(t_tokenizer *t, const char *token_str);

// Free tokenizer resources
void			tokenizer_free(t_tokenizer *t);

typedef struct s_merge_entry
{
	char					*pair;
	int						rank;
	struct s_merge_entry	*next;
}	t_merge_entry;

// Vocab hash map entry (zero-copy: token points to vocab[] string)
typedef struct s_vocab_entry
{
	char					*token;
	int						id;
	struct s_vocab_entry	*next;
}	t_vocab_entry;

# define VOCAB_HASH_SIZE 262147

typedef struct s_tokenizer_internal
{
	t_tokenizer		pub;
	t_merge_entry	**merge_map;
	t_vocab_entry	**vocab_map;
}	t_tokenizer_internal;

#endif
