/*
 * 42-BERLIN-ENGINE: Compute Operations Tests
 * ===========================================
 * Tests for compute/ops.c
 */

#include "test_harness.h"
#include "tensor/tensor.h"
#include "compute/ops.h"
#include "memory/arena.h"
#include <math.h>

/* ============================================
 * Helper: Create tensor with float data
 * ============================================ */
static t_tensor make_tensor(t_bf16 *buffer, int *shape, int ndim, float *values) {
    t_tensor t = tensor_view(buffer, shape, ndim);
    for (size_t i = 0; i < t.size; i++) {
        t.data[i] = float_to_bf16(values[i]);
    }
    return t;
}

/* ============================================
 * Test: RMSNorm Basic
 * ============================================ */
void test_rmsnorm_basic(void) {
    TEST_BEGIN("RMSNorm Basic");
    
    // Input: [1, 2, 3, 4]
    // RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) = 2.7386
    // Output = input / RMS * weight (weight=1)
    // Expected: [0.365, 0.730, 1.095, 1.460]
    
    t_bf16 buf_in[4], buf_out[4], buf_w[4];
    int shape[] = {1, 4};
    float in_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w_vals[] = {1.0f, 1.0f, 1.0f, 1.0f};
    
    t_tensor t_in = make_tensor(buf_in, shape, 2, in_vals);
    t_tensor t_w = make_tensor(buf_w, shape, 2, w_vals);
    t_tensor t_out = tensor_view(buf_out, shape, 2);
    
    op_rmsnorm(&t_out, &t_in, &t_w, 1e-5f);
    
    float expected_rms = sqrtf(7.5f);
    ASSERT_NEAR(1.0f / expected_rms, bf16_to_float(t_out.data[0]), BF16_TOLERANCE);
    ASSERT_NEAR(2.0f / expected_rms, bf16_to_float(t_out.data[1]), BF16_TOLERANCE);
    ASSERT_NEAR(3.0f / expected_rms, bf16_to_float(t_out.data[2]), BF16_TOLERANCE);
    ASSERT_NEAR(4.0f / expected_rms, bf16_to_float(t_out.data[3]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: RMSNorm with Weights
 * ============================================ */
void test_rmsnorm_weighted(void) {
    TEST_BEGIN("RMSNorm with Weights");
    
    t_bf16 buf_in[4], buf_out[4], buf_w[4];
    int shape[] = {1, 4};
    float in_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w_vals[] = {2.0f, 2.0f, 2.0f, 2.0f};  // Weight = 2
    
    t_tensor t_in = make_tensor(buf_in, shape, 2, in_vals);
    t_tensor t_w = make_tensor(buf_w, shape, 2, w_vals);
    t_tensor t_out = tensor_view(buf_out, shape, 2);
    
    op_rmsnorm(&t_out, &t_in, &t_w, 1e-5f);
    
    float expected_rms = sqrtf(7.5f);
    // Output scaled by weight=2. Use looser tolerance as larger values have more BF16 error
    ASSERT_NEAR(2.0f * 1.0f / expected_rms, bf16_to_float(t_out.data[0]), BF16_TOLERANCE_LOOSE);
    ASSERT_NEAR(2.0f * 4.0f / expected_rms, bf16_to_float(t_out.data[3]), BF16_TOLERANCE_LOOSE);
    
    TEST_END();
}

/* ============================================
 * Test: RMSNorm Zero Input
 * ============================================ */
void test_rmsnorm_zero(void) {
    TEST_BEGIN("RMSNorm Zero Input (epsilon safety)");
    
    t_bf16 buf_in[4], buf_out[4], buf_w[4];
    int shape[] = {1, 4};
    float in_vals[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float w_vals[] = {1.0f, 1.0f, 1.0f, 1.0f};
    
    t_tensor t_in = make_tensor(buf_in, shape, 2, in_vals);
    t_tensor t_w = make_tensor(buf_w, shape, 2, w_vals);
    t_tensor t_out = tensor_view(buf_out, shape, 2);
    
    op_rmsnorm(&t_out, &t_in, &t_w, 1e-5f);
    
    // Should not crash, output should be 0 or near-0
    ASSERT_FALSE(isnan(bf16_to_float(t_out.data[0])));
    ASSERT_FALSE(isinf(bf16_to_float(t_out.data[0])));
    
    TEST_END();
}

/* ============================================
 * Test: RoPE Position 0 (No rotation)
 * ============================================ */
void test_rope_pos_zero(void) {
    TEST_BEGIN("RoPE Position 0 (No Rotation)");
    
    t_bf16 buf[4];
    int shape[] = {1, 4};
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    t_tensor t = make_tensor(buf, shape, 2, vals);
    
    op_rope(&t, 0, 4, 10000.0f);
    
    // At position 0, cos(0)=1, sin(0)=0, so no rotation
    ASSERT_NEAR(1.0f, bf16_to_float(t.data[0]), BF16_TOLERANCE);
    ASSERT_NEAR(2.0f, bf16_to_float(t.data[1]), BF16_TOLERANCE);
    ASSERT_NEAR(3.0f, bf16_to_float(t.data[2]), BF16_TOLERANCE);
    ASSERT_NEAR(4.0f, bf16_to_float(t.data[3]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: RoPE Rotation at Position 1
 * ============================================ */
void test_rope_pos_one(void) {
    TEST_BEGIN("RoPE Position 1 (With Rotation)");
    
    t_bf16 buf[4];
    int shape[] = {1, 4};
    float vals[] = {1.0f, 0.0f, 1.0f, 0.0f};  // Orthogonal pairs
    
    t_tensor t = make_tensor(buf, shape, 2, vals);
    
    op_rope(&t, 1, 4, 10000.0f);
    
    // Check that values changed (rotated)
    float v0 = bf16_to_float(t.data[0]);
    float v1 = bf16_to_float(t.data[1]);
    
    // For pos=1, j=0: theta = 1/10000^0 = 1, angle = 1 radian
    // r0 = 1*cos(1) - 0*sin(1) = cos(1) ≈ 0.5403
    // r1 = 1*sin(1) + 0*cos(1) = sin(1) ≈ 0.8415
    ASSERT_NEAR(cosf(1.0f), v0, BF16_TOLERANCE);
    ASSERT_NEAR(sinf(1.0f), v1, BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: MatMul 2x2 Identity
 * ============================================ */
void test_matmul_identity(void) {
    TEST_BEGIN("MatMul with Identity");
    
    t_bf16 buf_a[4], buf_b[4], buf_out[4];
    int shape[] = {2, 2};
    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[] = {1.0f, 0.0f, 0.0f, 1.0f};  // Identity
    
    t_tensor t_a = make_tensor(buf_a, shape, 2, a_vals);
    t_tensor t_b = make_tensor(buf_b, shape, 2, b_vals);
    t_tensor t_out = tensor_view(buf_out, shape, 2);
    
    op_matmul(&t_out, &t_a, &t_b);
    
    // A * I = A
    ASSERT_NEAR(1.0f, bf16_to_float(t_out.data[0]), BF16_TOLERANCE);
    ASSERT_NEAR(2.0f, bf16_to_float(t_out.data[1]), BF16_TOLERANCE);
    ASSERT_NEAR(3.0f, bf16_to_float(t_out.data[2]), BF16_TOLERANCE);
    ASSERT_NEAR(4.0f, bf16_to_float(t_out.data[3]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: MatMul General
 * ============================================ */
void test_matmul_general(void) {
    TEST_BEGIN("MatMul General [2x3] x [3x2]");
    
    t_bf16 buf_a[6], buf_b[6], buf_out[4];
    int shape_a[] = {2, 3};
    int shape_b[] = {3, 2};
    int shape_out[] = {2, 2};
    
    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    t_tensor t_a = make_tensor(buf_a, shape_a, 2, a_vals);
    t_tensor t_b = make_tensor(buf_b, shape_b, 2, b_vals);
    t_tensor t_out = tensor_view(buf_out, shape_out, 2);
    
    op_matmul(&t_out, &t_a, &t_b);
    
    // out[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    // out[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    // out[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
    // out[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
    
    ASSERT_NEAR(22.0f, bf16_to_float(t_out.data[0]), BF16_TOLERANCE_LOOSE);
    ASSERT_NEAR(28.0f, bf16_to_float(t_out.data[1]), BF16_TOLERANCE_LOOSE);
    ASSERT_NEAR(49.0f, bf16_to_float(t_out.data[2]), BF16_TOLERANCE_LOOSE);
    ASSERT_NEAR(64.0f, bf16_to_float(t_out.data[3]), BF16_TOLERANCE_LOOSE);
    
    TEST_END();
}

/* ============================================
 * Test: Lightning Indexer Basic
 * ============================================ */
void test_lightning_score_basic(void) {
    TEST_BEGIN("Lightning Indexer Score");
    
    // Q: 2 heads, dim 2
    // Head 0: [1, 0], Weight 1.0
    // Head 1: [0, 1], Weight 0.5
    // K: 3 keys
    // Key 0: [1, 0] -> Dot0=1, Dot1=0 -> Score = 1*1 + 0.5*0 = 1.0
    // Key 1: [0, 1] -> Dot0=0, Dot1=1 -> Score = 1*0 + 0.5*1 = 0.5
    // Key 2: [1, 1] -> Dot0=1, Dot1=1 -> Score = 1*1 + 0.5*1 = 1.5
    
    t_bf16 buf_q[4], buf_k[6], buf_w[2], buf_scores[3];
    
    int shape_q[] = {2, 2};
    int shape_k[] = {3, 2};
    int shape_w[] = {2};
    int shape_scores[] = {3};
    
    float q_vals[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_vals[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    float w_vals[] = {1.0f, 0.5f};
    
    t_tensor t_q = make_tensor(buf_q, shape_q, 2, q_vals);
    t_tensor t_k = make_tensor(buf_k, shape_k, 2, k_vals);
    t_tensor t_w = make_tensor(buf_w, shape_w, 1, w_vals);
    t_tensor t_scores = tensor_view(buf_scores, shape_scores, 1);
    
    op_lightning_score(&t_scores, &t_q, &t_k, &t_w);
    
    ASSERT_NEAR(1.0f, bf16_to_float(t_scores.data[0]), BF16_TOLERANCE);
    ASSERT_NEAR(0.5f, bf16_to_float(t_scores.data[1]), BF16_TOLERANCE);
    ASSERT_NEAR(1.5f, bf16_to_float(t_scores.data[2]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: Lightning Indexer ReLU
 * ============================================ */
void test_lightning_score_relu(void) {
    TEST_BEGIN("Lightning Indexer ReLU (Negative Dot)");
    
    t_bf16 buf_q[2], buf_k[2], buf_w[1], buf_scores[1];
    
    int shape_q[] = {1, 2};
    int shape_k[] = {1, 2};
    int shape_w[] = {1};
    int shape_scores[] = {1};
    
    float q_vals[] = {1.0f, 0.0f};
    float k_vals[] = {-1.0f, 0.0f};  // Negative dot product
    float w_vals[] = {1.0f};
    
    t_tensor t_q = make_tensor(buf_q, shape_q, 2, q_vals);
    t_tensor t_k = make_tensor(buf_k, shape_k, 2, k_vals);
    t_tensor t_w = make_tensor(buf_w, shape_w, 1, w_vals);
    t_tensor t_scores = tensor_view(buf_scores, shape_scores, 1);
    
    op_lightning_score(&t_scores, &t_q, &t_k, &t_w);
    
    // Dot is -1, ReLU clamps to 0
    ASSERT_NEAR(0.0f, bf16_to_float(t_scores.data[0]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: Top-K Selection
 * ============================================ */
void test_topk_basic(void) {
    TEST_BEGIN("Top-K Selection (K=2)");
    
    t_arena scratch;
    arena_init(&scratch, 4096);
    
    t_bf16 buf[5];
    int shape[] = {5};
    float vals[] = {0.5f, 2.0f, 1.0f, 3.0f, 0.1f};
    
    t_tensor t = make_tensor(buf, shape, 1, vals);
    
    int indices[2];
    op_topk_select(indices, &t, 2, &scratch);
    
    // Top 2 are: 3.0 (index 3), 2.0 (index 1)
    ASSERT_EQ(3, indices[0]);
    ASSERT_EQ(1, indices[1]);
    
    arena_free(&scratch);
    TEST_END();
}

/* ============================================
 * Test: Top-K with K > N
 * ============================================ */
void test_topk_k_larger(void) {
    TEST_BEGIN("Top-K (K > N)");
    
    t_arena scratch;
    arena_init(&scratch, 4096);
    
    t_bf16 buf[3];
    int shape[] = {3};
    float vals[] = {1.0f, 2.0f, 3.0f};
    
    t_tensor t = make_tensor(buf, shape, 1, vals);
    
    int indices[5];
    op_topk_select(indices, &t, 5, &scratch);  // K=5 but only 3 elements
    
    // Should return all 3 in sorted order
    ASSERT_EQ(2, indices[0]);  // 3.0
    ASSERT_EQ(1, indices[1]);  // 2.0
    ASSERT_EQ(0, indices[2]);  // 1.0
    
    arena_free(&scratch);
    TEST_END();
}

/* ============================================
 * Test: SiLU Activation
 * ============================================ */
void test_silu_basic(void) {
    TEST_BEGIN("SiLU Activation");
    
    t_bf16 buf_gate[2], buf_val[2], buf_out[2];
    int shape[] = {2};
    
    // SiLU(x) = x * sigmoid(x)
    // SiLU(0) = 0
    // SiLU(2) = 2 / (1 + e^-2) = 2 / 1.135 ≈ 1.762
    float gate_vals[] = {0.0f, 2.0f};
    float val_vals[] = {1.0f, 1.0f};
    
    t_tensor t_gate = make_tensor(buf_gate, shape, 1, gate_vals);
    t_tensor t_val = make_tensor(buf_val, shape, 1, val_vals);
    t_tensor t_out = tensor_view(buf_out, shape, 1);
    
    op_silu_mul(&t_out, &t_gate, &t_val);
    
    // SiLU(0) * 1 = 0
    ASSERT_NEAR(0.0f, bf16_to_float(t_out.data[0]), BF16_TOLERANCE);
    
    // SiLU(2) * 1 ≈ 1.762
    float expected = 2.0f / (1.0f + expf(-2.0f));
    ASSERT_NEAR(expected, bf16_to_float(t_out.data[1]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: SwiGLU (SiLU * Val)
 * ============================================ */
void test_swiglu(void) {
    TEST_BEGIN("SwiGLU (SiLU * Val)");
    
    t_bf16 buf_gate[1], buf_val[1], buf_out[1];
    int shape[] = {1};
    
    float gate_vals[] = {2.0f};
    float val_vals[] = {3.0f};
    
    t_tensor t_gate = make_tensor(buf_gate, shape, 1, gate_vals);
    t_tensor t_val = make_tensor(buf_val, shape, 1, val_vals);
    t_tensor t_out = tensor_view(buf_out, shape, 1);
    
    op_silu_mul(&t_out, &t_gate, &t_val);
    
    // SiLU(2) * 3 ≈ 1.762 * 3 = 5.286
    float silu_2 = 2.0f / (1.0f + expf(-2.0f));
    ASSERT_NEAR(silu_2 * 3.0f, bf16_to_float(t_out.data[0]), BF16_TOLERANCE_LOOSE);
    
    TEST_END();
}

/* ============================================
 * Main
 * ============================================ */
int main(void) {
    TEST_SUITE_BEGIN("Compute Operations");
    
    test_rmsnorm_basic();
    test_rmsnorm_weighted();
    test_rmsnorm_zero();
    
    test_rope_pos_zero();
    test_rope_pos_one();
    
    test_matmul_identity();
    test_matmul_general();
    
    test_lightning_score_basic();
    test_lightning_score_relu();
    
    test_topk_basic();
    test_topk_k_larger();
    
    test_silu_basic();
    test_swiglu();
    
    TEST_SUITE_END();
}
