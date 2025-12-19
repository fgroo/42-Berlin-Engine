#include "inference/inference.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
	t_transformer t;
	
	if (argc < 3)
	{
		printf("Usage: %s <model_path> <config_path>\n", argv[0]);
		return (1);
	}

	printf("Initializing transformer...\n");
	if (transformer_init(&t, argv[1], argv[2]) != 0)
	{
		printf("Failed to init transformer\n");
		return (1);
	}
	printf("Transformer initialized. Layers: %d, Dim: %d\n", t.config.n_layers, t.config.dim);

	// Run forward pass with BOS token (1)
	printf("Running forward pass...\n");
	float *logits = transformer_forward(&t, 1, 0);
	
	printf("Forward pass complete.\n");
	
	// Check for NaNs
	int nans = 0;
	for (int i = 0; i < t.config.vocab_size; i++)
	{
		if (isnan(logits[i]))
		{
			nans++;
		}
	}
	printf("NaNs in logits: %d\n", nans);

	// Print top 5 logits
	printf("Top 5 logits:\n");
	for (int k = 0; k < 5; k++)
	{
		float max_val = -INFINITY;
		int max_idx = -1;
		for (int i = 0; i < t.config.vocab_size; i++)
		{
			if (logits[i] > max_val)
			{
				max_val = logits[i];
				max_idx = i;
			}
		}
		printf("%d: %f\n", max_idx, max_val);
		logits[max_idx] = -INFINITY; // Mask out
	}

	transformer_free(&t);
	return (0);
}
