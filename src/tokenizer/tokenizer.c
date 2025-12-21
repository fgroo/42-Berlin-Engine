/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tokenizer.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/05 14:25:00 by fgroo       #+#    #+#             */
/*   Updated: 2025/12/05 14:40:00 by fgroo      ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "tokenizer.h"
#include "../memory/safe_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <pthread.h>

#define MAX_LINE_LEN 4096
#define MAX_VOCAB_SIZE 132000
#define MERGE_HASH_SIZE 200003

// GPT-2 / Mistral byte-to-unicode reverse mapping
// Maps special unicode chars back to byte values (0x00-0xFF)
// This is the inverse of the bytes_to_unicode() mapping used by HuggingFace tokenizers
static int	g_byte_decoder[65536];
static pthread_once_t	g_byte_decoder_once = PTHREAD_ONCE_INIT;

static void	init_byte_decoder_impl(void)
{
	int	bs[256];
	int	cs[256];
	int	n_bs;
	int	n;
	int	b;
	int	i;

	// Initialize printable ASCII ranges that map to themselves
	n_bs = 0;
	for (b = '!'; b <= '~'; b++)
		bs[n_bs++] = b;
	for (b = 0xA1; b <= 0xAC; b++)  // ¡ to ¬
		bs[n_bs++] = b;
	for (b = 0xAE; b <= 0xFF; b++)  // ® to ÿ
		bs[n_bs++] = b;
	// Copy to cs (these chars represent themselves)
	for (i = 0; i < n_bs; i++)
		cs[i] = bs[i];
	// Non-printable bytes get mapped to 256+n
	n = 0;
	for (b = 0; b < 256; b++)
	{
		int found = 0;
		for (i = 0; i < n_bs; i++)
			if (bs[i] == b) { found = 1; break; }
		if (!found)
		{
			bs[n_bs] = b;
			cs[n_bs] = 256 + n;
			n_bs++;
			n++;
		}
	}
	// Build reverse mapping: unicode_char -> byte_value
	for (i = 0; i < 65536; i++)
		g_byte_decoder[i] = -1;  // -1 means not in mapping
	for (i = 0; i < n_bs; i++)
		g_byte_decoder[cs[i]] = bs[i];
}

static void	init_byte_decoder(void)
{
	pthread_once(&g_byte_decoder_once, init_byte_decoder_impl);
}

// Decode a unicode codepoint to its byte value (-1 if not mapped)
static int	decode_byte(int codepoint)
{
	if (codepoint < 0 || codepoint >= 65536)
		return (-1);
	return (g_byte_decoder[codepoint]);
}

static unsigned int	hash_str(const char *str)
{
	unsigned int	hash;
	int				c;

	hash = 5381;
	while ((c = *str++))
		hash = ((hash << 5) + hash) + c;
	return (hash);
}

static char	*json_string_decode(const char *s, int len)
{
	char	*res;
	int		i;
	int		j;

	/*
	** FIX Issue #3: Buffer Overflow Prevention
	** UTF-16 surrogate pairs can expand to 4-byte UTF-8 sequences.
	** Worst case: each input char becomes 4 output bytes.
	** Example: \uD83D\uDE00 (12 chars) -> 😀 (4 bytes)
	*/
	res = xmalloc(len * 4 + 1);
	i = 0;
	j = 0;
	while (i < len)
	{
		if (s[i] == '\\')
		{
			i++;
			if (i >= len)
				break ;
			if (s[i] == 'u' && i + 4 < len)
			{
				// Decode \uXXXX to UTF-8
				unsigned int codepoint = 0;
				sscanf(&s[i + 1], "%4x", &codepoint);
				i += 5; // Skip uXXXX
				
				// Handle UTF-16 surrogate pairs for emoji (codepoints > 0xFFFF)
				// High surrogate: 0xD800-0xDBFF, Low surrogate: 0xDC00-0xDFFF
				if (codepoint >= 0xD800 && codepoint <= 0xDBFF)
				{
					// Check for low surrogate \uXXXX following
					if (s[i] == '\\' && s[i + 1] == 'u' && i + 5 < len)
					{
						unsigned int low = 0;
						sscanf(&s[i + 2], "%4x", &low);
						if (low >= 0xDC00 && low <= 0xDFFF)
						{
							// Combine surrogates: ((high - 0xD800) << 10) + (low - 0xDC00) + 0x10000
							codepoint = ((codepoint - 0xD800) << 10) + (low - 0xDC00) + 0x10000;
							i += 6; // Skip \uXXXX
						}
					}
				}
				
				// Convert codepoint to UTF-8
				if (codepoint < 0x80) {
					res[j++] = (char)codepoint;
				} else if (codepoint < 0x800) {
					res[j++] = (char)(0xC0 | (codepoint >> 6));
					res[j++] = (char)(0x80 | (codepoint & 0x3F));
				} else if (codepoint < 0x10000) {
					res[j++] = (char)(0xE0 | (codepoint >> 12));
					res[j++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
					res[j++] = (char)(0x80 | (codepoint & 0x3F));
				} else {
					// 4-byte UTF-8 for codepoints >= 0x10000 (emojis)
					res[j++] = (char)(0xF0 | (codepoint >> 18));
					res[j++] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
					res[j++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
					res[j++] = (char)(0x80 | (codepoint & 0x3F));
				}
				continue;
			}
			else if (s[i] == 'n')
				res[j++] = '\n';
			else if (s[i] == 't')
				res[j++] = '\t';
			else if (s[i] == 'r')
				res[j++] = '\r';
			else
				res[j++] = s[i];
		}
		else
		{
			res[j++] = s[i];
		}
		i++;
	}
	res[j] = '\0';
	return (res);
}


static void	add_merge(t_tokenizer_internal *ti, char *pair, int rank)
{
	unsigned int	h;
	t_merge_entry	*entry;

	h = hash_str(pair) % MERGE_HASH_SIZE;
	entry = xmalloc(sizeof(t_merge_entry));
	entry->pair = strdup(pair);
	entry->rank = rank;
	entry->next = ti->merge_map[h];
	ti->merge_map[h] = entry;
}

static int	get_merge_rank(t_tokenizer_internal *ti, const char *pair)
{
	unsigned int	h;
	t_merge_entry	*e;

	h = hash_str(pair) % MERGE_HASH_SIZE;
	e = ti->merge_map[h];
	while (e)
	{
		if (strcmp(e->pair, pair) == 0)
			return (e->rank);
		e = e->next;
	}
	return (-1);
}

/*
** Add token to vocab hash map (zero-copy: points to vocab[] string)
*/
static void	add_vocab_hash(t_tokenizer_internal *ti, char *token, int id)
{
	unsigned int	h;
	t_vocab_entry	*entry;
	t_vocab_entry	*e;

	h = hash_str(token) % VOCAB_HASH_SIZE;
	// Check if token already exists - if so, keep first occurrence (correct ID)
	e = ti->vocab_map[h];
	while (e)
	{
		if (strcmp(e->token, token) == 0)
		{
			// Token already exists - skip to preserve first ID
			return;
		}
		e = e->next;
	}
	entry = xmalloc(sizeof(t_vocab_entry));
	entry->token = token;
	entry->id = id;
	entry->next = ti->vocab_map[h];
	ti->vocab_map[h] = entry;
}



static int	is_vocab_end(const char *line)
{
	// Check if line is "    }," or similar
	while (*line && isspace(*line))
		line++;
	if (line[0] == '}' && line[1] == ',')
		return (1);
	if (line[0] == '}' && line[1] == '\0') // End of file case
		return (1);
	return (0);
}

static void	parse_vocab(t_tokenizer *t, FILE *f)
{
	char					line[MAX_LINE_LEN];
	char					*key_start;
	char					*key_end;
	char					*val_start;
	int						id;
	char					*token;
	t_tokenizer_internal	*ti;

	ti = (t_tokenizer_internal *)t->priv;
	t->vocab = xcalloc(MAX_VOCAB_SIZE, sizeof(char *));
	t->vocab_size = 0;
	while (fgets(line, sizeof(line), f))
	{
		if (is_vocab_end(line))
			break ;
		key_start = strchr(line, '"');
		if (!key_start)
			continue ;
		key_start++;
		val_start = strchr(key_start, ':');
		if (!val_start)
			continue ;
		key_end = val_start - 1;
		while (key_end > key_start && *key_end != '"')
			key_end--;
		if (key_end <= key_start)
			continue ;
		id = atoi(val_start + 1);
		if (id >= MAX_VOCAB_SIZE)
			continue ;
		if (id >= t->vocab_size)
			t->vocab_size = id + 1;
		token = json_string_decode(key_start, key_end - key_start);
		t->vocab[id] = token;
		add_vocab_hash(ti, token, id);
	}
}

static void	parse_merges(t_tokenizer_internal *ti, FILE *f)
{
	char	line[MAX_LINE_LEN];
	char	*p;
	char	*start;
	char	*end;
	char	first[256];
	char	second[256];
	char	pair[512];
	int		rank;
	int		state; // 0: wait [, 1: read 1st, 2: read 2nd, 3: wait ]

	rank = 0;
	state = 0;
	while (fgets(line, sizeof(line), f))
	{
		p = line;
		// Skip initial whitespace
		while (*p && isspace(*p)) p++;
		
		if (strstr(p, "\"merges\": [")) continue;
		
		// Check for end of merges array
		if (state == 0 && *p == ']' && (*(p+1) == ',' || *(p+1) == '\0' || isspace(*(p+1))))
			break;

		while (*p)
		{
			if (state == 0)
			{
				if (*p == '[')
				{
					state = 1;
				}
				p++;
			}
			else if (state == 1)
			{
				if (*p == '"')
				{
					start = p + 1;
					end = strchr(start, '"');
					if (end)
					{
						char *dec = json_string_decode(start, end - start);
						strncpy(first, dec, 255);
						first[255] = 0;
						free(dec);
						state = 2;
						p = end + 1;
					}
					else
						break; // Should not happen if valid json line
				}
				else
					p++;
			}
			else if (state == 2)
			{
				if (*p == '"')
				{
					start = p + 1;
					end = strchr(start, '"');
					if (end)
					{
						char *dec = json_string_decode(start, end - start);
						strncpy(second, dec, 255);
						second[255] = 0;
						free(dec);
						
						snprintf(pair, sizeof(pair), "%s %s", first, second);
						add_merge(ti, pair, rank);
						rank++;
						state = 3;
						p = end + 1;
					}
					else
						break;
				}
				else
					p++;
			}
			else if (state == 3)
			{
				if (*p == ']')
				{
					state = 0;
				}
				p++;
			}
		}
	}
}

int	tokenizer_init(t_tokenizer *t, const char *json_path)
{
	FILE					*f;
	char					line[MAX_LINE_LEN];
	t_tokenizer_internal	*ti;

	// printf("Tokenizer init: %s\n", json_path);
	init_byte_decoder();  // Initialize GPT-2 byte decoder table
	f = fopen(json_path, "r");
	if (!f)
		return (-1);
	
	t->vocab = NULL;
	t->priv = NULL;
	t->bos_id = 1;
	t->eos_id = 2;
	t->unk_id = 0;

	ti = xcalloc(1, sizeof(t_tokenizer_internal));
	ti->merge_map = xcalloc(MERGE_HASH_SIZE, sizeof(t_merge_entry *));
	ti->vocab_map = xcalloc(VOCAB_HASH_SIZE, sizeof(t_vocab_entry *));
	t->priv = ti;

	while (fgets(line, sizeof(line), f))
	{
		if (strstr(line, "\"vocab\": {"))
		{
			// printf("Parsing vocab...\n");
			parse_vocab(t, f);
		}
		else if (strstr(line, "\"merges\": ["))
		{
			// printf("Parsing merges...\n");
			parse_merges(ti, f);
		}
	}
	fclose(f);
	
	if (t->vocab_size == 0)
	{
		fprintf(stderr, "Error: No vocab found in %s. Is it a valid tokenizer.json?\n", json_path);
		/* [FIX] Free allocated memory before returning error */
		if (ti->merge_map)
			free(ti->merge_map);
		if (ti->vocab_map)
			free(ti->vocab_map);
		free(ti);
		t->priv = NULL;
		return (1);
	}
	
	// printf("Tokenizer init done\n");
	return (0);
}

// O(1) hash lookup for token ID
static int	get_token_id(t_tokenizer *t, const char *token)
{
	t_tokenizer_internal	*ti;
	unsigned int			h;
	t_vocab_entry			*e;

	ti = (t_tokenizer_internal *)t->priv;
	h = hash_str(token) % VOCAB_HASH_SIZE;
	e = ti->vocab_map[h];
	while (e)
	{
		if (strcmp(e->token, token) == 0)
			return (e->id);
		e = e->next;
	}
	return (t->unk_id);
}

typedef struct s_bpe_token
{
	char				*str;
	struct s_bpe_token	*next;
	struct s_bpe_token	*prev;
	int					version;  // For invalidation tracking
}	t_bpe_token;

/*
** Heap entry for BPE merge prioritization
** We use a min-heap where lower rank = higher priority
*/
typedef struct s_bpe_heap_item
{
	int			rank;		// Merge rank (lower = merge first)
	t_bpe_token	*left;		// Left token of the pair
	int			left_ver;	// Version when this entry was created
	int			right_ver;	// Version of right token when created
}	t_bpe_heap_item;

#define BPE_HEAP_MAX 65536

static void	bpe_heap_sift_up(t_bpe_heap_item *heap, int idx)
{
	t_bpe_heap_item	tmp;
	int				parent;

	while (idx > 0)
	{
		parent = (idx - 1) / 2;
		if (heap[parent].rank <= heap[idx].rank)
			break ;
		tmp = heap[parent];
		heap[parent] = heap[idx];
		heap[idx] = tmp;
		idx = parent;
	}
}

static void	bpe_heap_sift_down(t_bpe_heap_item *heap, int size, int idx)
{
	t_bpe_heap_item	tmp;
	int				smallest;
	int				left;
	int				right;

	while (1)
	{
		smallest = idx;
		left = 2 * idx + 1;
		right = 2 * idx + 2;
		if (left < size && heap[left].rank < heap[smallest].rank)
			smallest = left;
		if (right < size && heap[right].rank < heap[smallest].rank)
			smallest = right;
		if (smallest == idx)
			break ;
		tmp = heap[idx];
		heap[idx] = heap[smallest];
		heap[smallest] = tmp;
		idx = smallest;
	}
}

static void	bpe_heap_push(t_bpe_heap_item *heap, int *size, int rank,
				t_bpe_token *left)
{
	if (*size >= BPE_HEAP_MAX)
		return ;
	heap[*size].rank = rank;
	heap[*size].left = left;
	heap[*size].left_ver = left->version;
	heap[*size].right_ver = left->next ? left->next->version : -1;
	bpe_heap_sift_up(heap, *size);
	(*size)++;
}

static t_bpe_heap_item	bpe_heap_pop(t_bpe_heap_item *heap, int *size)
{
	t_bpe_heap_item	top;

	top = heap[0];
	(*size)--;
	if (*size > 0)
	{
		heap[0] = heap[*size];
		bpe_heap_sift_down(heap, *size, 0);
	}
	return (top);
}

static void	free_bpe_list(t_bpe_token *head)
{
	t_bpe_token	*tmp;

	while (head)
	{
		tmp = head;
		head = head->next;
		free(tmp->str);
		free(tmp);
	}
}

/*
** O(N log N) BPE Encoding using Min-Heap
** 
** Algorithm:
** 1. Split text into characters, build doubly-linked list
** 2. Compute merge rank for ALL adjacent pairs, push to min-heap
** 3. Pop best pair from heap (O(log N))
** 4. If pair is still valid (versions match), merge it
** 5. Push new pairs (left neighbor + merged, merged + right neighbor)
** 6. Repeat until heap empty or no valid merges
*/
int	tokenizer_encode(t_tokenizer *t, const char *text, int **tokens)
{
	t_tokenizer_internal	*ti;
	t_bpe_token				*head;
	t_bpe_token				*curr;
	t_bpe_token				*last;
	t_bpe_token				*deleted_head;  /* Trash list for deferred cleanup */
	t_bpe_heap_item			*heap;
	int						heap_size;
	int						i;
	int						len;
	char					pair_buf[512];
	int						rank;
	t_bpe_heap_item			item;
	t_bpe_token				*left;
	t_bpe_token				*right;
	char					*new_str;

	ti = (t_tokenizer_internal *)t->priv;
	head = NULL;
	last = NULL;
	deleted_head = NULL;
	heap = xmalloc(BPE_HEAP_MAX * sizeof(t_bpe_heap_item));
	heap_size = 0;

	// 1. Initial split into characters (same as before)
	i = 0;
	while (text[i])
	{
		curr = xcalloc(1, sizeof(t_bpe_token));
		curr->version = 0;
		if (text[i] == ' ')
		{
			curr->str = xmalloc(3);
			strcpy(curr->str, "Ġ");
		}
		else
		{
			curr->str = xmalloc(2);
			curr->str[0] = text[i];
			curr->str[1] = '\0';
		}
		curr->prev = last;
		if (last)
			last->next = curr;
		else
			head = curr;
		last = curr;
		i++;
	}

	// 2. Build initial heap with all adjacent pairs O(N log N)
	curr = head;
	while (curr && curr->next)
	{
		snprintf(pair_buf, sizeof(pair_buf), "%s %s", curr->str, curr->next->str);
		rank = get_merge_rank(ti, pair_buf);
		if (rank >= 0)
			bpe_heap_push(heap, &heap_size, rank, curr);
		curr = curr->next;
	}

	// 3. Process heap until empty
	while (heap_size > 0)
	{
		item = bpe_heap_pop(heap, &heap_size);
		left = item.left;
		
		// Validate: check if left token is still valid (not deleted)
		if (!left->str)
			continue ;  // Token was deleted in a previous merge
		
		// Validate: check if this pair still exists (versions match)
		if (!left->next)
			continue ;
		if (!left->next->str)
			continue ;  // Right token was deleted
		if (left->version != item.left_ver)
			continue ;
		if (left->next->version != item.right_ver)
			continue ;
		
		right = left->next;
		
		// Merge: left absorbs right
		new_str = xmalloc(strlen(left->str) + strlen(right->str) + 1);
		strcpy(new_str, left->str);
		strcat(new_str, right->str);
		free(left->str);
		left->str = new_str;
		left->version++;  // Invalidate old heap entries for this token
		
		// Mark right as deleted (lazy deletion - keeps pointers valid)
		free(right->str);
		right->str = NULL;  // Mark as deleted
		
		// Unlink right from list
		left->next = right->next;
		if (right->next)
			right->next->prev = left;
		
		/*
		** FIX Issue #1: BPE Memory Leak - Deferred Cleanup
		** We can't free(right) immediately because old heap entries may still
		** reference this pointer via item.left. The validation at line 625
		** checks if (!left->str) which would be a dangling pointer access.
		**
		** Solution: Link deleted nodes into a "trash" list for cleanup.
		** We reuse right->next to chain them (right->prev is now stale).
		*/
		right->next = deleted_head;
		deleted_head = right;
		
		// Push new pairs involving merged token
		// Left neighbor + merged token
		if (left->prev)
		{
			snprintf(pair_buf, sizeof(pair_buf), "%s %s", 
				left->prev->str, left->str);
			rank = get_merge_rank(ti, pair_buf);
			if (rank >= 0)
				bpe_heap_push(heap, &heap_size, rank, left->prev);
		}
		// Merged token + right neighbor
		if (left->next)
		{
			snprintf(pair_buf, sizeof(pair_buf), "%s %s", 
				left->str, left->next->str);
			rank = get_merge_rank(ti, pair_buf);
			if (rank >= 0)
				bpe_heap_push(heap, &heap_size, rank, left);
		}
	}

	free(heap);
	
	/* FIX Issue #1: Free all deleted nodes now that heap is done */
	while (deleted_head)
	{
		t_bpe_token	*tmp = deleted_head;
		deleted_head = deleted_head->next;
		free(tmp);  /* str already freed above */
	}

	// 4. Convert to IDs
	len = 0;
	curr = head;
	while (curr)
	{
		len++;
		curr = curr->next;
	}

	*tokens = xmalloc(len * sizeof(int));
	i = 0;
	curr = head;
	while (curr)
	{
		(*tokens)[i++] = get_token_id(t, curr->str);
		curr = curr->next;
	}
	
	free_bpe_list(head);
	return (len);
}

// Thread-local buffer for decoded tokens (handles Ġ → space conversion)
// THREAD-SAFE: Each thread gets its own buffer, no race conditions with OpenMP
static __thread char	g_decode_buf[256];

const char	*tokenizer_decode(t_tokenizer *t, int token_id)
{
	const char		*raw;
	unsigned char	*out;
	int				i;
	int				j;
	int				codepoint;
	int				byte_val;

	if (token_id < 0 || token_id >= t->vocab_size)
		return ("");
	if (!t->vocab[token_id])
		return ("");
	
	raw = t->vocab[token_id];
	out = (unsigned char *)g_decode_buf;
	i = 0;
	j = 0;
	
	// Handle special tokens that should not be printed
	if (raw[0] == '<' && (strstr(raw, "<s>") || strstr(raw, "</s>") || 
		strstr(raw, "[INST]") || strstr(raw, "[/INST]")))
		return ("");
	
	// Handle hex byte tokens: <0xHH> -> actual byte value
	if (raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>')
	{
		unsigned int bv = 0;
		sscanf(raw + 3, "%2x", &bv);
		out[0] = (unsigned char)bv;
		out[1] = '\0';
		return (g_decode_buf);
	}
	
	// Decode using GPT-2 byte mapping
	// Each UTF-8 encoded codepoint in raw maps to one byte in output
	while (raw[i] && j < 254)
	{
		unsigned char c = (unsigned char)raw[i];
		
		// Decode UTF-8 to get the Unicode codepoint
		if ((c & 0x80) == 0)
		{
			// ASCII: check if it's a mapped byte
			codepoint = c;
			i++;
		}
		else if ((c & 0xE0) == 0xC0)
		{
			// 2-byte UTF-8
			codepoint = ((c & 0x1F) << 6) | ((unsigned char)raw[i+1] & 0x3F);
			i += 2;
		}
		else if ((c & 0xF0) == 0xE0)
		{
			// 3-byte UTF-8
			codepoint = ((c & 0x0F) << 12) |
				(((unsigned char)raw[i+1] & 0x3F) << 6) |
				((unsigned char)raw[i+2] & 0x3F);
			i += 3;
		}
		else if ((c & 0xF8) == 0xF0)
		{
			// 4-byte UTF-8
			codepoint = ((c & 0x07) << 18) |
				(((unsigned char)raw[i+1] & 0x3F) << 12) |
				(((unsigned char)raw[i+2] & 0x3F) << 6) |
				((unsigned char)raw[i+3] & 0x3F);
			i += 4;
		}
		else
		{
			// Invalid, copy as-is
			out[j++] = c;
			i++;
			continue;
		}
		
		// Look up in byte decoder
		byte_val = decode_byte(codepoint);
		if (byte_val >= 0)
		{
			// GPT-2 decoder returns raw byte values that form UTF-8 sequences
			// Multiple tokens combine to create complete UTF-8 characters
			out[j++] = (unsigned char)byte_val;
		}
		else
		{
			// Not in byte mapping, encode codepoint back to UTF-8
			if (codepoint < 0x80)
				out[j++] = (unsigned char)codepoint;
			else if (codepoint < 0x800)
			{
				out[j++] = 0xC0 | (codepoint >> 6);
				out[j++] = 0x80 | (codepoint & 0x3F);
			}
			else if (codepoint < 0x10000)
			{
				out[j++] = 0xE0 | (codepoint >> 12);
				out[j++] = 0x80 | ((codepoint >> 6) & 0x3F);
				out[j++] = 0x80 | (codepoint & 0x3F);
			}
			else
			{
				out[j++] = 0xF0 | (codepoint >> 18);
				out[j++] = 0x80 | ((codepoint >> 12) & 0x3F);
				out[j++] = 0x80 | ((codepoint >> 6) & 0x3F);
				out[j++] = 0x80 | (codepoint & 0x3F);
			}
		}
	}
	out[j] = '\0';
	return (g_decode_buf);
}

void	tokenizer_free(t_tokenizer *t)
{
	int						i;
	t_tokenizer_internal	*ti;
	t_merge_entry			*me;
	t_merge_entry			*me_next;
	t_vocab_entry			*ve;
	t_vocab_entry			*ve_next;

	ti = (t_tokenizer_internal *)t->priv;
	if (t->vocab)
	{
		i = 0;
		while (i < t->vocab_size)
		{
			free(t->vocab[i]);
			i++;
		}
		free(t->vocab);
	}
	if (ti)
	{
		i = 0;
		while (i < MERGE_HASH_SIZE)
		{
			me = ti->merge_map[i];
			while (me)
			{
				me_next = me->next;
				free(me->pair);
				free(me);
				me = me_next;
			}
			i++;
		}
		free(ti->merge_map);
		i = 0;
		while (i < VOCAB_HASH_SIZE)
		{
			ve = ti->vocab_map[i];
			while (ve)
			{
				ve_next = ve->next;
				free(ve);
				ve = ve_next;
			}
			i++;
		}
		free(ti->vocab_map);
		free(ti);
	}
}

/*
** MOPD: Public lookup function for token string -> ID
** Used by /v1/distill endpoint to convert teacher's token strings
** to local token IDs for proper distillation.
*/
int	tokenizer_lookup_id(t_tokenizer *t, const char *token_str)
{
	return (get_token_id(t, token_str));
}
