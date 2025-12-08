#include "loader/loader.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    t_model model;
    if (load_model(&model, argv[1]) != 0) {
        printf("Failed to load model\n");
        return 1;
    }

    printf("Loaded %d tensors:\n", model.num_tensors);
    for (int i = 0; i < model.num_tensors; i++) {
        t_tensor *t = &model.tensors[i].tensor;
        printf("%s: [", model.tensors[i].name);
        for (int j = 0; j < t->ndim; j++) {
            printf("%d", t->shape[j]);
            if (j < t->ndim - 1) printf("x");
        }
        printf("]\n");
    }

    free_model(&model);
    return 0;
}
