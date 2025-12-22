#include <stdio.h>
#include "loader/loader.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s model.safetensors\n", argv[0]);
        return 1;
    }
    
    t_model model;
    if (load_model(&model, argv[1]) != 0) {
        printf("Failed to load model\n");
        return 1;
    }
    
    printf("Found %d tensors:\n", model.num_tensors);
    for (int i = 0; i < model.num_tensors && i < 500; i++) {
        printf("  [%d] %s (shape: ", i, model.tensors[i].name);
        for (int d = 0; d < model.tensors[i].tensor.ndim; d++) {
            printf("%d%s", model.tensors[i].tensor.shape[d], 
                   d < model.tensors[i].tensor.ndim - 1 ? " x " : "");
        }
        printf(")\n");
    }
    
    free_model(&model);
    return 0;
}
