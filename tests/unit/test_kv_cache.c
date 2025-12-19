/*
 * 42-BERLIN-ENGINE: KV Cache Tests
 * ==================================
 * Tests for memory/kv_cache.c
 * 
 * NOTE: Updated for new API with init_params and evict_params structs
 */

#include "test_harness.h"
#include "memory/arena.h"
#include "memory/kv_cache.h"
#include "tensor/tensor.h"
#include "compute/ops.h"

/* ============================================
 * Helper: Create tensor with float data
 * ============================================ */
static t_tensor make_tensor(t_bf16 *buffer, int *shape, int ndim, float *values) {
    t_tensor t = tensor_view(buffer, shape, ndim);
    for (size_t i = 0; i < t.size; i++) {
        ((t_bf16*)t.data)[i] = float_to_bf16(values[i]);
    }
    return t;
}

/* ============================================
 * Test: KV Cache Initialization
 * ============================================ */
void test_kv_cache_init(void) {
    TEST_BEGIN("KV Cache Initialization");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    t_kv_cache cache;
    t_kv_init_params params = {
        .max_seq = 16,
        .num_heads = 4,
        .head_dim = 8
    };
    kv_cache_init(&cache, &arena, &params);
    
    ASSERT_EQ(16, cache.max_seq_len);
    ASSERT_EQ(4, cache.num_heads);
    ASSERT_EQ(8, cache.head_dim);
    ASSERT_EQ(0, cache.current_seq_len);
    ASSERT_NOT_NULL(cache.k.data);
    ASSERT_NOT_NULL(cache.v.data);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: KV Cache Append
 * ============================================ */
void test_kv_cache_append(void) {
    TEST_BEGIN("KV Cache Append");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    t_kv_cache cache;
    t_kv_init_params params = {
        .max_seq = 8,
        .num_heads = 1,
        .head_dim = 2
    };
    kv_cache_init(&cache, &arena, &params);
    
    t_bf16 buf_k[2], buf_v[2];
    int shape[] = {1, 2};
    float k_vals[] = {1.0f, 2.0f};
    float v_vals[] = {3.0f, 4.0f};
    
    t_tensor t_k = make_tensor(buf_k, shape, 2, k_vals);
    t_tensor t_v = make_tensor(buf_v, shape, 2, v_vals);
    
    kv_cache_append(&cache, &t_k, &t_v);
    
    ASSERT_EQ(1, cache.current_seq_len);
    
    // Verify K was stored
    ASSERT_NEAR(1.0f, bf16_to_float(((t_bf16*)cache.k.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(2.0f, bf16_to_float(((t_bf16*)cache.k.data)[1]), BF16_TOLERANCE);
    
    // Verify V was stored
    ASSERT_NEAR(3.0f, bf16_to_float(((t_bf16*)cache.v.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(4.0f, bf16_to_float(((t_bf16*)cache.v.data)[1]), BF16_TOLERANCE);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: KV Cache Multiple Appends
 * ============================================ */
void test_kv_cache_multiple_appends(void) {
    TEST_BEGIN("KV Cache Multiple Appends");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    t_kv_cache cache;
    t_kv_init_params params = {
        .max_seq = 8,
        .num_heads = 1,
        .head_dim = 1
    };
    kv_cache_init(&cache, &arena, &params);
    
    t_bf16 buf_k[1], buf_v[1];
    int shape[] = {1, 1};
    
    for (int i = 1; i <= 4; i++) {
        float k_val = (float)i;
        float v_val = (float)(i * 10);
        
        t_tensor t_k = make_tensor(buf_k, shape, 2, &k_val);
        t_tensor t_v = make_tensor(buf_v, shape, 2, &v_val);
        
        kv_cache_append(&cache, &t_k, &t_v);
    }
    
    ASSERT_EQ(4, cache.current_seq_len);
    
    // Verify sequence order
    ASSERT_NEAR(1.0f, bf16_to_float(((t_bf16*)cache.k.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(2.0f, bf16_to_float(((t_bf16*)cache.k.data)[1]), BF16_TOLERANCE);
    ASSERT_NEAR(3.0f, bf16_to_float(((t_bf16*)cache.k.data)[2]), BF16_TOLERANCE);
    ASSERT_NEAR(4.0f, bf16_to_float(((t_bf16*)cache.k.data)[3]), BF16_TOLERANCE);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: KV Cache Full Warning
 * ============================================ */
void test_kv_cache_full(void) {
    TEST_BEGIN("KV Cache Full (No Overflow)");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    t_kv_cache cache;
    t_kv_init_params params = {
        .max_seq = 4,
        .num_heads = 1,
        .head_dim = 1
    };
    kv_cache_init(&cache, &arena, &params);
    
    t_bf16 buf_k[1], buf_v[1];
    int shape[] = {1, 1};
    float val = 1.0f;
    
    t_tensor t_k = make_tensor(buf_k, shape, 2, &val);
    t_tensor t_v = make_tensor(buf_v, shape, 2, &val);
    
    // Fill cache
    for (int i = 0; i < 4; i++) {
        kv_cache_append(&cache, &t_k, &t_v);
    }
    ASSERT_EQ(4, cache.current_seq_len);
    
    // Try to append when full (should not increment)
    kv_cache_append(&cache, &t_k, &t_v);
    ASSERT_EQ(4, cache.current_seq_len);  // Still 4, not 5
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: KV Cache Multi-Head
 * ============================================ */
void test_kv_cache_multihead(void) {
    TEST_BEGIN("KV Cache Multi-Head");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    t_kv_cache cache;
    t_kv_init_params params = {
        .max_seq = 8,
        .num_heads = 2,
        .head_dim = 4
    };
    kv_cache_init(&cache, &arena, &params);
    
    // K/V shape: [num_heads=2, head_dim=4] = 8 elements
    t_bf16 buf_k[8], buf_v[8];
    int shape[] = {2, 4};
    float k_vals[] = {1,2,3,4, 5,6,7,8};  // Head 0: 1-4, Head 1: 5-8
    float v_vals[] = {1,1,1,1, 2,2,2,2};
    
    t_tensor t_k = make_tensor(buf_k, shape, 2, k_vals);
    t_tensor t_v = make_tensor(buf_v, shape, 2, v_vals);
    
    kv_cache_append(&cache, &t_k, &t_v);
    
    ASSERT_EQ(1, cache.current_seq_len);
    
    // Verify layout: cache.k should be [seq, heads, dim]
    ASSERT_NEAR(1.0f, bf16_to_float(((t_bf16*)cache.k.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(5.0f, bf16_to_float(((t_bf16*)cache.k.data)[4]), BF16_TOLERANCE);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: KV Cache Eviction Basic
 * ============================================ */
void test_kv_cache_evict_basic(void) {
    TEST_BEGIN("KV Cache Eviction (Keep Top-2)");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    t_kv_cache cache;
    t_kv_init_params init_params = {
        .max_seq = 8,
        .num_heads = 1,
        .head_dim = 1
    };
    kv_cache_init(&cache, &arena, &init_params);
    
    t_bf16 buf_k[1], buf_v[1];
    int shape[] = {1, 1};
    
    // Append 4 tokens with K values: 1, 2, 3, 4
    for (int i = 1; i <= 4; i++) {
        float k_val = (float)i;
        float v_val = (float)i;
        
        t_tensor t_k = make_tensor(buf_k, shape, 2, &k_val);
        t_tensor t_v = make_tensor(buf_v, shape, 2, &v_val);
        
        kv_cache_append(&cache, &t_k, &t_v);
    }
    ASSERT_EQ(4, cache.current_seq_len);
    
    // Evict to keep top-2
    t_bf16 buf_q[1], buf_w[1];
    float q_val = 1.0f;
    float w_val = 1.0f;
    
    t_tensor t_q = make_tensor(buf_q, shape, 2, &q_val);
    int shape_w[] = {1};
    t_tensor t_w = make_tensor(buf_w, shape_w, 1, &w_val);
    
    t_evict_params evict_params = {
        .q = &t_q,
        .w = &t_w,
        .keep_k = 2,
        .scratch = &arena
    };
    kv_cache_evict(&cache, &evict_params);
    
    ASSERT_EQ(2, cache.current_seq_len);
    
    // Should have kept tokens with K=3 and K=4 (in order)
    ASSERT_NEAR(3.0f, bf16_to_float(((t_bf16*)cache.k.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(4.0f, bf16_to_float(((t_bf16*)cache.k.data)[1]), BF16_TOLERANCE);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Main
 * ============================================ */
int main(void) {
    TEST_SUITE_BEGIN("KV Cache");
    
    test_kv_cache_init();
    test_kv_cache_append();
    test_kv_cache_multiple_appends();
    test_kv_cache_full();
    test_kv_cache_multihead();
    test_kv_cache_evict_basic();
    
    TEST_SUITE_END();
}
