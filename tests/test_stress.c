/*
 * 42-BERLIN-ENGINE: Stress Tests
 * ================================
 * Memory pressure, large tensors, and edge cases.
 */

#include "test_harness.h"
#include "memory/arena.h"
#include "memory/kv_cache.h"
#include "tensor/tensor.h"
#include "compute/ops.h"
#include "compute/sampler.h"
#include <time.h>
#include <stdlib.h>

/* ============================================
 * Test: Large t_arena Allocation
 * ============================================ */
void test_stress_large_arena(void) {
    TEST_BEGIN("Stress: Large t_arena (256MB)");
    
    size_t size = 256 * 1024 * 1024;  // 256MB
    
    t_arena arena;
    arena_init(&arena, size);
    ASSERT_NOT_NULL(arena.base);
    
    // Allocate in chunks
    int num_allocs = 100;
    size_t chunk_size = 2 * 1024 * 1024;  // 2MB chunks
    
    for (int i = 0; i < num_allocs; i++) {
        void *ptr = arena_alloc(&arena, chunk_size);
        ASSERT_NOT_NULL(ptr);
    }
    
    LOG_INFO("Allocated %d x 2MB chunks = %zuMB", num_allocs, (num_allocs * chunk_size) / (1024*1024));
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Many Small Allocations
 * ============================================ */
void test_stress_many_small_allocs(void) {
    TEST_BEGIN("Stress: 10000 Small Allocations");
    
    t_arena arena;
    arena_init(&arena, 64 * 1024 * 1024);  // 64MB
    
    int count = 10000;
    
    for (int i = 0; i < count; i++) {
        void *ptr = arena_alloc(&arena, 64);  // 64 bytes each
        ASSERT_NOT_NULL(ptr);
        
        // Write to memory to ensure it's accessible
        ((uint8_t*)ptr)[0] = (uint8_t)(i & 0xFF);
    }
    
    LOG_INFO("Successfully allocated %d x 64 bytes", count);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Large MatMul
 * ============================================ */
void test_stress_large_matmul(void) {
    TEST_BEGIN("Stress: Large MatMul [256x256] x [256x256]");
    
    t_arena arena;
    arena_init(&arena, 16 * 1024 * 1024);  // 16MB
    
    int M = 256, K = 256, N = 256;
    
    t_bf16 *a_data = arena_alloc(&arena, M * K * sizeof(t_bf16));
    t_bf16 *b_data = arena_alloc(&arena, K * N * sizeof(t_bf16));
    t_bf16 *c_data = arena_alloc(&arena, M * N * sizeof(t_bf16));
    
    // Initialize with random-ish values
    for (int i = 0; i < M * K; i++) {
        a_data[i] = float_to_bf16((float)(i % 10) * 0.1f);
    }
    for (int i = 0; i < K * N; i++) {
        b_data[i] = float_to_bf16((float)(i % 10) * 0.1f);
    }
    
    int shape_a[] = {M, K};
    int shape_b[] = {K, N};
    int shape_c[] = {M, N};
    
    t_tensor t_a = tensor_view(a_data, shape_a, 2);
    t_tensor t_b = tensor_view(b_data, shape_b, 2);
    t_tensor t_c = tensor_view(c_data, shape_c, 2);
    
    clock_t start = clock();
    op_matmul(&t_c, &t_a, &t_b);
    clock_t end = clock();
    
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    LOG_INFO("MatMul completed in %.2f ms", time_ms);
    
    // Sanity check - result should not be all zeros
    int non_zero = 0;
    for (int i = 0; i < 100; i++) {
        if (bf16_to_float(t_c.data[i]) != 0.0f) non_zero++;
    }
    ASSERT_TRUE(non_zero > 0);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Large Top-K
 * ============================================ */
void test_stress_large_topk(void) {
    TEST_BEGIN("Stress: Top-K on 10000 Elements");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    int n = 10000;
    int k = 100;
    
    t_bf16 *scores = arena_alloc(&arena, n * sizeof(t_bf16));
    int *indices = arena_alloc(&arena, k * sizeof(int));
    
    // Descending values (10000, 9999, 9998, ...)
    for (int i = 0; i < n; i++) {
        scores[i] = float_to_bf16((float)(n - i));
    }
    
    int shape[] = {n};
    t_tensor t_scores = tensor_view(scores, shape, 1);
    
    clock_t start = clock();
    op_topk_select(indices, &t_scores, k, &arena);
    clock_t end = clock();
    
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    LOG_INFO("Top-%d selection from %d elements in %.2f ms", k, n, time_ms);
    
    // Top-K should be indices 0, 1, 2, ... (highest scores)
    ASSERT_EQ(0, indices[0]);
    ASSERT_EQ(1, indices[1]);
    ASSERT_EQ(99, indices[99]);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: KV Cache Rapid Eviction Cycles
 * ============================================ */
void test_stress_kv_eviction_cycles(void) {
    TEST_BEGIN("Stress: KV Cache Eviction Cycles");
    
    t_arena arena;
    arena_init(&arena, 64 * 1024 * 1024);
    
    t_kv_cache cache;
    kv_cache_init(&cache, &arena, 100, 2, 8);  // 100 tokens, 2 heads, dim 8
    
    t_bf16 buf_k[16], buf_v[16];
    int shape[] = {2, 8};
    
    // Fill with initial tokens
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 16; j++) {
            buf_k[j] = float_to_bf16((float)i * 0.01f);
            buf_v[j] = float_to_bf16((float)i * 0.02f);
        }
        t_tensor t_k = tensor_view(buf_k, shape, 2);
        t_tensor t_v = tensor_view(buf_v, shape, 2);
        kv_cache_append(&cache, &t_k, &t_v);
    }
    ASSERT_EQ(100, cache.current_seq_len);
    
    // Perform multiple eviction cycles
    t_bf16 buf_q[16], buf_w[2];
    for (int j = 0; j < 16; j++) buf_q[j] = float_to_bf16(1.0f);
    buf_w[0] = float_to_bf16(1.0f);
    buf_w[1] = float_to_bf16(1.0f);
    
    t_tensor t_q = tensor_view(buf_q, shape, 2);
    int shape_w[] = {2};
    t_tensor t_w = tensor_view(buf_w, shape_w, 1);
    
    int num_cycles = 10;
    for (int cycle = 0; cycle < num_cycles; cycle++) {
        // Add 10 new tokens
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 16; j++) {
                buf_k[j] = float_to_bf16((float)(100 + cycle * 10 + i) * 0.01f);
                buf_v[j] = float_to_bf16((float)(100 + cycle * 10 + i) * 0.02f);
            }
            t_tensor t_k = tensor_view(buf_k, shape, 2);
            t_tensor t_v = tensor_view(buf_v, shape, 2);
            
            if (cache.current_seq_len < cache.max_seq_len) {
                kv_cache_append(&cache, &t_k, &t_v);
            }
        }
        
        // Evict back to 50
        if (cache.current_seq_len > 50) {
            kv_cache_evict(&cache, &t_q, &t_w, 50, &arena);
        }
    }
    
    LOG_INFO("Completed %d eviction cycles, final seq_len = %d", num_cycles, cache.current_seq_len);
    ASSERT_TRUE(cache.current_seq_len <= 100);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: RMSNorm Large Batch
 * ============================================ */
void test_stress_rmsnorm_batch(void) {
    TEST_BEGIN("Stress: RMSNorm Batch [128 x 512]");
    
    t_arena arena;
    arena_init(&arena, 16 * 1024 * 1024);
    
    int batch = 128;
    int dim = 512;
    
    t_bf16 *input = arena_alloc(&arena, batch * dim * sizeof(t_bf16));
    t_bf16 *output = arena_alloc(&arena, batch * dim * sizeof(t_bf16));
    t_bf16 *weight = arena_alloc(&arena, dim * sizeof(t_bf16));
    
    // Initialize
    for (int i = 0; i < batch * dim; i++) {
        input[i] = float_to_bf16((float)(i % 100) * 0.05f);
    }
    for (int i = 0; i < dim; i++) {
        weight[i] = float_to_bf16(1.0f);
    }
    
    int shape_in[] = {batch, dim};
    int shape_w[] = {1, dim};
    
    t_tensor t_in = tensor_view(input, shape_in, 2);
    t_tensor t_out = tensor_view(output, shape_in, 2);
    t_tensor t_w = tensor_view(weight, shape_w, 2);
    
    clock_t start = clock();
    op_rmsnorm(&t_out, &t_in, &t_w, 1e-5f);
    clock_t end = clock();
    
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    LOG_INFO("RMSNorm on [%d x %d] completed in %.2f ms", batch, dim, time_ms);
    
    // Check first row normalization
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = bf16_to_float(t_out.data[i]);
        sum_sq += val * val;
    }
    float rms = sqrtf(sum_sq / dim);
    ASSERT_NEAR(1.0f, rms, 0.2f);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: RoPE Many Positions
 * ============================================ */
void test_stress_rope_positions(void) {
    TEST_BEGIN("Stress: RoPE at Positions 0-1000");
    
    t_bf16 buffer[64];
    int shape[] = {1, 64};
    
    for (int pos = 0; pos < 1000; pos += 100) {
        // Initialize fresh each time
        for (int i = 0; i < 64; i++) {
            buffer[i] = float_to_bf16(1.0f);
        }
        
        t_tensor t = tensor_view(buffer, shape, 2);
        op_rope(&t, pos, 64, 10000.0f);
        
        // Check no NaN/Inf
        for (int i = 0; i < 64; i++) {
            float val = bf16_to_float(t.data[i]);
            ASSERT_FALSE(isnan(val));
            ASSERT_FALSE(isinf(val));
        }
    }
    
    LOG_INFO("RoPE tested at positions 0, 100, 200, ..., 900");
    TEST_END();
}

/* ============================================
 * Test: t_arena Reset Stress
 * ============================================ */
void test_stress_arena_reset(void) {
    TEST_BEGIN("Stress: t_arena Reset Cycles");
    
    t_arena arena;
    arena_init(&arena, 10 * 1024 * 1024);  // 10MB
    
    int cycles = 100;
    
    for (int c = 0; c < cycles; c++) {
        // Allocate bunch of stuff
        for (int i = 0; i < 100; i++) {
            void *ptr = arena_alloc(&arena, 8192);
            ASSERT_NOT_NULL(ptr);
        }
        
        // Reset
        arena_reset(&arena);
        ASSERT_EQ(0, arena.offset);
    }
    
    LOG_INFO("Completed %d arena reset cycles", cycles);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: BF16 Edge Cases Mass
 * ============================================ */
void test_stress_bf16_edge_cases(void) {
    TEST_BEGIN("Stress: BF16 Edge Cases");
    
    // Very small values
    for (int i = 0; i < 1000; i++) {
        float val = (float)i * 1e-6f;
        t_bf16 bf = float_to_bf16(val);
        float recovered = bf16_to_float(bf);
        ASSERT_FALSE(isnan(recovered));
    }
    
    // Very large values
    for (int i = 0; i < 1000; i++) {
        float val = (float)i * 100.0f;
        t_bf16 bf = float_to_bf16(val);
        float recovered = bf16_to_float(bf);
        ASSERT_FALSE(isnan(recovered));
    }
    
    // Negative values
    for (int i = 0; i < 1000; i++) {
        float val = -(float)i * 0.5f;
        t_bf16 bf = float_to_bf16(val);
        float recovered = bf16_to_float(bf);
        ASSERT_FALSE(isnan(recovered));
        ASSERT_TRUE(recovered <= 0.0f);
    }
    
    LOG_INFO("Tested 3000 BF16 edge cases");
    TEST_END();
}

/* ============================================
 * Main
 * ============================================ */
int main(void) {
    TEST_SUITE_BEGIN("Stress Tests");
    
    test_stress_large_arena();
    test_stress_many_small_allocs();
    test_stress_large_matmul();
    test_stress_large_topk();
    test_stress_kv_eviction_cycles();
    test_stress_rmsnorm_batch();
    test_stress_rope_positions();
    test_stress_arena_reset();
    test_stress_bf16_edge_cases();
    
    TEST_SUITE_END();
}
