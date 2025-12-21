/* Tokenizer Probe - Diagnose tokenization issues */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer/tokenizer.h"

int main(int argc, char **argv)
{
    t_tokenizer tok;
    int         *tokens;
    int         n_tokens;
    int         i;
    const char  *piece;

    if (argc < 3) {
        printf("Usage: %s <tokenizer.json> <text>\n", argv[0]);
        printf("Example: %s Ministral-Stuff/tokenizer.json \"Hello world\"\n", argv[0]);
        return 1;
    }

    printf("=== TOKENIZER PROBE ===\n");
    printf("Loading: %s\n", argv[1]);

    if (tokenizer_init(&tok, argv[1]) != 0) {
        printf("[FAIL] Could not load tokenizer\n");
        return 1;
    }
    printf("[OK] Vocab size: %d\n", tok.vocab_size);

    printf("\nInput text: '%s'\n", argv[2]);
    printf("Input length: %zu bytes\n\n", strlen(argv[2]));

    tokens = NULL;
    n_tokens = tokenizer_encode(&tok, argv[2], &tokens);

    printf("=== RESULT ===\n");
    printf("Token count: %d\n", n_tokens);
    
    if (n_tokens <= 0 || !tokens) {
        printf("[FAIL] Tokenization returned %d tokens (NULL=%d)\n", 
               n_tokens, tokens == NULL);
        return 1;
    }

    printf("Token IDs: [ ");
    for (i = 0; i < n_tokens && i < 50; i++) {
        printf("%d ", tokens[i]);
    }
    if (n_tokens > 50) printf("...");
    printf("]\n\n");

    printf("=== DECODE CHECK ===\n");
    for (i = 0; i < n_tokens && i < 20; i++) {
        piece = tokenizer_decode(&tok, tokens[i]);
        printf("  [%d] %d -> '%s'\n", i, tokens[i], piece ? piece : "(null)");
    }
    if (n_tokens > 20) printf("  ... (%d more tokens)\n", n_tokens - 20);

    free(tokens);
    tokenizer_free(&tok);
    
    printf("\n=== DIAGNOSIS ===\n");
    if (n_tokens <= 2) {
        printf("[WARNING] Very few tokens! Possible issues:\n");
        printf("  - Empty input string\n");
        printf("  - Tokenizer format mismatch\n");
        printf("  - All chars mapped to UNK\n");
    } else {
        printf("[OK] Token count looks reasonable\n");
    }
    
    return 0;
}
