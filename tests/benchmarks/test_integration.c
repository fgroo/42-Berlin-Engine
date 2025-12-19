/*
 * 42-BERLIN-ENGINE: Integration Tests
 * =====================================
 * End-to-end testing of the complete inference pipeline.
 */

#include "../test_harness.h"
#include "memory/arena.h"
#include "memory/kv_cache.h"
#include "tensor/tensor.h"
#include "compute/ops.h"
#include "compute/sampler.h"
#include "nested/fluid.h"
#include "loader/loader.h"
#include <string.h>

/* ============================================
 * Helper: Create tensor with float data
 * ============================================ */
static t_tensor make_tensor(t_bf16 *buffer, int *shape, int ndim, float *values) {
    t_tensor t = tensor_view(buffer, shape, ndim);
    if (values) {
        for (size_t i = 0; i < t.size; i++) {
            ((t_bf16*)t.data)[i] = float_to_bf16(values[i]);
        }
    }
    return t;
}

/* ============================================
 * Test: t_arena + t_tensor Integration
 * ============================================ */
void test_arena_tensor_integration(void) {
    TEST_BEGIN("t_arena + t_tensor Integration");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    int shape[] = {4, 8};  // 32 elements
    size_t size = 32 * sizeof(t_bf16);
    
    void *data = arena_alloc(&arena, size);
    ASSERT_NOT_NULL(data);
    
    t_tensor t = tensor_view(data, shape, 2);
    
    // Write and read back
    for (int i = 0; i < 32; i++) {
        ((t_bf16*)t.data)[i] = float_to_bf16((float)i);
    }
    
    ASSERT_NEAR(0.0f, bf16_to_float(((t_bf16*)t.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(15.0f, bf16_to_float(((t_bf16*)t.data)[15]), BF16_TOLERANCE);
    ASSERT_NEAR(31.0f, bf16_to_float(((t_bf16*)t.data)[31]), BF16_TOLERANCE);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Forward Pass Simulation
 * ============================================ */
void test_forward_pass_simple(void) {
    TEST_BEGIN("Simple Forward Pass");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    // Simulate: input -> RMSNorm -> Linear -> SwiGLU -> Output
    
    int dim = 4;
    int shape[] = {1, dim};
    
    // Input
    t_bf16 *input_buf = arena_alloc(&arena, dim * sizeof(t_bf16));
    float input_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    t_tensor input = make_tensor(input_buf, shape, 2, input_vals);
    
    // RMSNorm weights (all 1s)
    t_bf16 *norm_w_buf = arena_alloc(&arena, dim * sizeof(t_bf16));
    float norm_w_vals[] = {1.0f, 1.0f, 1.0f, 1.0f};
    t_tensor norm_w = make_tensor(norm_w_buf, shape, 2, norm_w_vals);
    
    // RMSNorm output
    t_bf16 *normed_buf = arena_alloc(&arena, dim * sizeof(t_bf16));
    t_tensor normed = tensor_view(normed_buf, shape, 2);
    
    op_rmsnorm(&normed, &input, &norm_w, 1e-5f);
    
    // Check normalized (should sum to approximately RMS = 1)
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = bf16_to_float(((t_bf16*)normed.data)[i]);
        sum_sq += val * val;
    }
    float rms = sqrtf(sum_sq / dim);
    ASSERT_NEAR(1.0f, rms, 0.1f);  // Should be ~1 after normalization
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: KV Cache with Eviction Pipeline
 * ============================================ */
void test_kv_cache_eviction_pipeline(void) {
    TEST_BEGIN("KV Cache Eviction Pipeline");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    t_kv_cache cache;
    kv_cache_init(&cache, &arena, 10, 1, 2);  // Max 10 tokens, 1 head, dim 2
    
    t_bf16 buf_k[2], buf_v[2];
    int shape[] = {1, 2};
    
    // Insert 10 tokens with increasing relevance
    for (int i = 0; i < 10; i++) {
        float k_vals[] = {(float)i, (float)i};
        float v_vals[] = {(float)(i * 10), (float)(i * 10)};
        
        t_tensor t_k = make_tensor(buf_k, shape, 2, k_vals);
        t_tensor t_v = make_tensor(buf_v, shape, 2, v_vals);
        
        kv_cache_append(&cache, &t_k, &t_v);
    }
    ASSERT_EQ(10, cache.current_seq_len);
    
    // Evict to keep only 5
    t_bf16 buf_q[2], buf_w[1];
    float q_vals[] = {1.0f, 1.0f};
    float w_vals[] = {1.0f};
    
    t_tensor t_q = make_tensor(buf_q, shape, 2, q_vals);
    int shape_w[] = {1};
    t_tensor t_w = make_tensor(buf_w, shape_w, 1, w_vals);
    
    kv_cache_evict(&cache, &t_q, &t_w, 5, &arena);
    
    ASSERT_EQ(5, cache.current_seq_len);
    
    // Top 5 should be tokens 5-9 (k values 5-9 have highest dot products)
    // After reordering by index: should be 5,6,7,8,9 in positions 0-4
    for (int i = 0; i < 5; i++) {
        float expected_k0 = (float)(5 + i);
        ASSERT_NEAR(expected_k0, bf16_to_float(((t_bf16*)cache.k.data)[i * 2]), BF16_TOLERANCE_LOOSE);
    }
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Attention with Sparse Selection
 * ============================================ */
void test_sparse_attention_flow(void) {
    TEST_BEGIN("Sparse Attention Flow (Lightning + TopK)");
    
    // Q: 1 head, dim 2
    // K: 5 keys, dim 2
    // Select Top-2 keys via Lightning Indexer
    
    t_bf16 buf_q[2], buf_k[10], buf_w[1], buf_scores[5];
    
    int shape_q[] = {1, 2};
    int shape_k[] = {5, 2};
    int shape_w[] = {1};
    int shape_scores[] = {5};
    
    // Q = [1, 0] - looking for keys with high first component
    float q_vals[] = {1.0f, 0.0f};
    
    // K = [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.5, 0.5]]
    // Dots with Q: 0.1, 0.8, 0.3, 0.9, 0.5
    // Top-2: index 3 (0.9), index 1 (0.8)
    float k_vals[] = {0.1f, 0.9f, 0.8f, 0.2f, 0.3f, 0.7f, 0.9f, 0.1f, 0.5f, 0.5f};
    
    float w_vals[] = {1.0f};
    
    t_tensor t_q = make_tensor(buf_q, shape_q, 2, q_vals);
    t_tensor t_k = make_tensor(buf_k, shape_k, 2, k_vals);
    t_tensor t_w = make_tensor(buf_w, shape_w, 1, w_vals);
    t_tensor t_scores = tensor_view(buf_scores, shape_scores, 1);
    
    op_lightning_score(&t_scores, &t_q, &t_k, &t_w);
    
    // Verify scores
    ASSERT_NEAR(0.1f, bf16_to_float(((t_bf16*)t_scores.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(0.8f, bf16_to_float(((t_bf16*)t_scores.data)[1]), BF16_TOLERANCE);
    ASSERT_NEAR(0.3f, bf16_to_float(((t_bf16*)t_scores.data)[2]), BF16_TOLERANCE);
    ASSERT_NEAR(0.9f, bf16_to_float(((t_bf16*)t_scores.data)[3]), BF16_TOLERANCE);
    ASSERT_NEAR(0.5f, bf16_to_float(((t_bf16*)t_scores.data)[4]), BF16_TOLERANCE);
    
    // Top-K selection (need a scratch arena)
    t_arena scratch;
    arena_init(&scratch, 4096);
    
    int indices[2];
    op_topk_select(indices, &t_scores, 2, &scratch);
    
    ASSERT_EQ(3, indices[0]);  // Highest: 0.9
    ASSERT_EQ(1, indices[1]);  // Second: 0.8
    
    arena_free(&scratch);
    TEST_END();
}

/* ============================================
 * Test: Nested Learning Training Loop
 * ============================================ */
void test_nested_learning_loop(void) {
    TEST_BEGIN("Nested Learning Training Loop");
    
    // Simulate: Forward -> Backward -> Update
    // y = x @ w, target = 1.0
    // x = [1.0], w = [0.0], lr = 0.5
    // After 10 steps, w should approach 1.0
    
    t_arena arena;
    arena_init(&arena, 4096);
    
    t_bf16 *x_buf = arena_alloc(&arena, sizeof(t_bf16));
    t_bf16 *w_buf = arena_alloc(&arena, sizeof(t_bf16));
    t_bf16 *grad_buf = arena_alloc(&arena, sizeof(t_bf16));
    t_bf16 *out_buf = arena_alloc(&arena, sizeof(t_bf16));
    
    int shape[] = {1, 1};
    
    t_tensor t_x = tensor_view(x_buf, shape, 2);
    ((t_bf16*)t_x.data)[0] = float_to_bf16(1.0f);
    
    t_tensor t_w = tensor_view(w_buf, shape, 2);
    ((t_bf16*)t_w.data)[0] = float_to_bf16(0.0f);  // Start at 0
    
    t_tensor t_grad = tensor_view(grad_buf, shape, 2);
    t_tensor t_out = tensor_view(out_buf, shape, 2);
    
    t_fluid_param fp = { &t_w, &t_grad };
    
    float target = 1.0f;
    float lr = 0.5f;
    
    for (int step = 0; step < 10; step++) {
        // Forward: y = x * w
        float x = bf16_to_float(((t_bf16*)t_x.data)[0]);
        float w = bf16_to_float(((t_bf16*)t_w.data)[0]);
        float y = x * w;
        
        // Loss gradient: dL/dy = y - target
        float grad_y = y - target;
        
        // Backward for y = x * w: dy/dw = x, so dL/dw = dL/dy * x = grad_y * x
        float grad_w = grad_y * x;
        ((t_bf16*)t_grad.data)[0] = float_to_bf16(grad_w);
        
        // Update
        optimizer_sgd(&fp, lr);
    }
    
    float final_w = bf16_to_float(((t_bf16*)t_w.data)[0]);
    LOG_INFO("Final weight after 10 steps: %.4f (target: 1.0)", final_w);
    
    ASSERT_NEAR(1.0f, final_w, 0.1f);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Sampler Integration
 * ============================================ */
void test_sampler_integration(void) {
    TEST_BEGIN("Argmax Sampler Integration");
    
    t_bf16 buf[100];
    int shape[] = {100};  // Vocab size 100
    
    t_tensor logits = tensor_view(buf, shape, 1);
    
    // Set all to low values
    for (int i = 0; i < 100; i++) {
        ((t_bf16*)logits.data)[i] = float_to_bf16(-1.0f);
    }
    
    // Set token 42 as winner
    ((t_bf16*)logits.data)[42] = float_to_bf16(10.0f);
    
    int token = sample_argmax(&logits);
    ASSERT_EQ(42, token);
    
    TEST_END();
}

/* ============================================
 * Test: Full Inference Step Simulation
 * ============================================ */
void test_full_inference_step(void) {
    TEST_BEGIN("Full Inference Step Simulation");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024);
    
    int seq_len = 8;
    int num_heads = 2;
    int head_dim = 4;
    int hidden_dim = num_heads * head_dim;  // 8
    
    // Initialize KV Cache
    t_kv_cache kv_cache;
    kv_cache_init(&kv_cache, &arena, 16, num_heads, head_dim);
    
    // Simulate processing 3 tokens
    for (int t = 0; t < 3; t++) {
        // Allocate input embedding
        int shape_hidden[] = {1, hidden_dim};
        t_bf16 *input = arena_alloc(&arena, hidden_dim * sizeof(t_bf16));
        for (int i = 0; i < hidden_dim; i++) {
            input[i] = float_to_bf16((float)(t + i) * 0.1f);
        }
        t_tensor t_input = tensor_view(input, shape_hidden, 2);
        
        // RMSNorm
        t_bf16 *norm_w = arena_alloc(&arena, hidden_dim * sizeof(t_bf16));
        t_bf16 *normed = arena_alloc(&arena, hidden_dim * sizeof(t_bf16));
        for (int i = 0; i < hidden_dim; i++) {
            norm_w[i] = float_to_bf16(1.0f);
        }
        t_tensor t_norm_w = tensor_view(norm_w, shape_hidden, 2);
        t_tensor t_normed = tensor_view(normed, shape_hidden, 2);
        
        op_rmsnorm(&t_normed, &t_input, &t_norm_w, 1e-5f);
        
        // Create K, V for this token (simplified: just use normed as both K and V)
        int shape_kv[] = {num_heads, head_dim};
        t_tensor t_k = tensor_view(normed, shape_kv, 2);
        t_tensor t_v = tensor_view(normed, shape_kv, 2);
        
        // Append to cache
        kv_cache_append(&kv_cache, &t_k, &t_v);
        
        // Apply RoPE
        op_rope(&t_k, t, head_dim, 10000.0f);
    }
    
    ASSERT_EQ(3, kv_cache.current_seq_len);
    LOG_INFO("Processed 3 tokens, cache seq_len = %d", kv_cache.current_seq_len);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Main
 * ============================================ */
int main(void) {
    TEST_SUITE_BEGIN("Integration Tests");
    
    test_arena_tensor_integration();
    test_forward_pass_simple();
    test_kv_cache_eviction_pipeline();
    test_sparse_attention_flow();
    test_nested_learning_loop();
    test_sampler_integration();
    test_full_inference_step();
    
    TEST_SUITE_END();
}
