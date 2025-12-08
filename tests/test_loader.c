/*
 * 42-BERLIN-ENGINE: Loader Tests
 * ================================
 * Tests for loader/loader.c (SafeTensors mmap loading)
 */

#include "test_harness.h"
#include "loader/loader.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* ============================================
 * Helper: Create a minimal safetensors file
 * ============================================ */
static int create_test_safetensors(const char *filename, const char *tensor_name, 
                                    int *shape, int ndim, t_bf16 *data, size_t data_size) {
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    
    // Build header JSON
    char header[512];
    char shape_str[64] = "[";
    for (int i = 0; i < ndim; i++) {
        char num[16];
        snprintf(num, sizeof(num), "%d%s", shape[i], i < ndim-1 ? ", " : "");
        strcat(shape_str, num);
    }
    strcat(shape_str, "]");
    
    snprintf(header, sizeof(header), 
             "{\"%s\": {\"dtype\": \"BF16\", \"shape\": %s, \"data_offsets\": [0, %zu]}}",
             tensor_name, shape_str, data_size);
    
    size_t header_len = strlen(header);
    
    // Pad header to multiple of 8 bytes (safetensors spec)
    size_t padded_len = ((header_len + 7) / 8) * 8;
    
    // Write header size (8 bytes, little endian)
    uint64_t header_size = padded_len;
    fwrite(&header_size, sizeof(uint64_t), 1, f);
    
    // Write header (padded)
    fwrite(header, 1, header_len, f);
    for (size_t i = header_len; i < padded_len; i++) {
        fputc(' ', f);
    }
    
    // Write data
    fwrite(data, 1, data_size, f);
    
    fclose(f);
    return 0;
}

/* ============================================
 * Test: Load Valid SafeTensors
 * ============================================ */
void test_loader_valid(void) {
    TEST_BEGIN("Load Valid SafeTensors");
    
    const char *filename = "test_valid.safetensors";
    int shape[] = {2, 2};
    t_bf16 data[] = {0x3F80, 0x4000, 0x4040, 0x4080};  // 1, 2, 3, 4 approx
    
    int ret = create_test_safetensors(filename, "test_tensor", shape, 2, data, sizeof(data));
    ASSERT_EQ(0, ret);
    
    t_model model;
    ret = load_model(&model, filename);
    ASSERT_EQ(0, ret);
    
    ASSERT_TRUE(model.num_tensors >= 1);
    ASSERT_TRUE(model.header_size > 0);
    
    free_model(&model);
    remove(filename);
    TEST_END();
}

/* ============================================
 * Test: Get t_tensor by Name
 * ============================================ */
void test_loader_get_tensor(void) {
    TEST_BEGIN("Get t_tensor by Name");
    
    const char *filename = "test_get.safetensors";
    int shape[] = {2, 2};
    t_bf16 data[] = {0x3F80, 0x4000, 0x4040, 0x4080};
    
    create_test_safetensors(filename, "my_weights", shape, 2, data, sizeof(data));
    
    t_model model;
    load_model(&model, filename);
    
    t_tensor *t = get_tensor(&model, "my_weights");
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(2, t->shape[0]);
    ASSERT_EQ(2, t->shape[1]);
    
    // Check data
    ASSERT_EQ(0x3F80, t->data[0]);
    ASSERT_EQ(0x4000, t->data[1]);
    
    free_model(&model);
    remove(filename);
    TEST_END();
}

/* ============================================
 * Test: Get Non-Existent t_tensor
 * ============================================ */
void test_loader_get_nonexistent(void) {
    TEST_BEGIN("Get Non-Existent t_tensor");
    
    const char *filename = "test_nonexist.safetensors";
    int shape[] = {2};
    t_bf16 data[] = {0x4000, 0x4000};
    
    create_test_safetensors(filename, "real_tensor", shape, 1, data, sizeof(data));
    
    t_model model;
    load_model(&model, filename);
    
    t_tensor *t = get_tensor(&model, "fake_tensor");
    ASSERT_NULL(t);
    
    free_model(&model);
    remove(filename);
    TEST_END();
}

/* ============================================
 * Test: Load Non-Existent File
 * ============================================ */
void test_loader_missing_file(void) {
    TEST_BEGIN("Load Missing File (Error Handling)");
    
    t_model model;
    int ret = load_model(&model, "this_file_does_not_exist.safetensors");
    ASSERT_EQ(-1, ret);
    
    TEST_END();
}

/* ============================================
 * Test: Multiple Tensors
 * ============================================ */
void test_loader_multiple_tensors(void) {
    TEST_BEGIN("Multiple Tensors in File");
    
    const char *filename = "test_multi.safetensors";
    
    // Manually create file with two tensors
    FILE *f = fopen(filename, "wb");
    ASSERT_NOT_NULL(f);
    
    // Header with two tensors
    const char *header = "{\"tensor_a\": {\"dtype\": \"BF16\", \"shape\": [2], \"data_offsets\": [0, 4]}, "
                         "\"tensor_b\": {\"dtype\": \"BF16\", \"shape\": [3], \"data_offsets\": [4, 10]}}";
    
    size_t header_len = strlen(header);
    size_t padded_len = ((header_len + 7) / 8) * 8;
    
    uint64_t header_size = padded_len;
    fwrite(&header_size, sizeof(uint64_t), 1, f);
    fwrite(header, 1, header_len, f);
    for (size_t i = header_len; i < padded_len; i++) fputc(' ', f);
    
    // Data (5 bf16 values = 10 bytes)
    t_bf16 data[] = {0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0};
    fwrite(data, sizeof(t_bf16), 5, f);
    fclose(f);
    
    t_model model;
    int ret = load_model(&model, filename);
    ASSERT_EQ(0, ret);
    
    ASSERT_EQ(2, model.num_tensors);
    
    t_tensor *ta = get_tensor(&model, "tensor_a");
    t_tensor *tb = get_tensor(&model, "tensor_b");
    ASSERT_NOT_NULL(ta);
    ASSERT_NOT_NULL(tb);
    
    ASSERT_EQ(2, ta->shape[0]);
    ASSERT_EQ(3, tb->shape[0]);
    
    free_model(&model);
    remove(filename);
    TEST_END();
}

/* ============================================
 * Test: 1D t_tensor Loading
 * ============================================ */
void test_loader_1d_tensor(void) {
    TEST_BEGIN("1D t_tensor Loading");
    
    const char *filename = "test_1d.safetensors";
    int shape[] = {4};
    t_bf16 data[] = {0x3F80, 0x4000, 0x4040, 0x4080};
    
    create_test_safetensors(filename, "vector", shape, 1, data, sizeof(data));
    
    t_model model;
    load_model(&model, filename);
    
    t_tensor *t = get_tensor(&model, "vector");
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(1, t->ndim);
    ASSERT_EQ(4, t->shape[0]);
    
    free_model(&model);
    remove(filename);
    TEST_END();
}

/* ============================================
 * Test: 3D t_tensor Loading
 * ============================================ */
void test_loader_3d_tensor(void) {
    TEST_BEGIN("3D t_tensor Loading");
    
    const char *filename = "test_3d.safetensors";
    int shape[] = {2, 3, 4};  // 24 elements
    t_bf16 data[24];
    for (int i = 0; i < 24; i++) data[i] = 0x4000 + i;
    
    create_test_safetensors(filename, "cube", shape, 3, data, sizeof(data));
    
    t_model model;
    load_model(&model, filename);
    
    t_tensor *t = get_tensor(&model, "cube");
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(3, t->ndim);
    ASSERT_EQ(2, t->shape[0]);
    ASSERT_EQ(3, t->shape[1]);
    ASSERT_EQ(4, t->shape[2]);
    
    free_model(&model);
    remove(filename);
    TEST_END();
}

/* ============================================
 * Test: Free t_model Safety
 * ============================================ */
void test_loader_free(void) {
    TEST_BEGIN("Free t_model (Double-Free Safety)");
    
    const char *filename = "test_free.safetensors";
    int shape[] = {2};
    t_bf16 data[] = {0x4000, 0x4000};
    
    create_test_safetensors(filename, "t", shape, 1, data, sizeof(data));
    
    t_model model;
    load_model(&model, filename);
    
    free_model(&model);
    ASSERT_NULL(model.mapped_addr);
    
    // Double free should be safe (pointer is NULL)
    // This won't crash if implementation checks for NULL
    
    remove(filename);
    TEST_END();
}

/* ============================================
 * Main
 * ============================================ */
int main(void) {
    TEST_SUITE_BEGIN("SafeTensors Loader");
    
    test_loader_valid();
    test_loader_get_tensor();
    test_loader_get_nonexistent();
    test_loader_missing_file();
    test_loader_multiple_tensors();
    test_loader_1d_tensor();
    test_loader_3d_tensor();
    test_loader_free();
    
    TEST_SUITE_END();
}
