/*
 * 42-BERLIN-ENGINE: Nested Learning (Fluid Weights) Tests
 * =========================================================
 * Tests for nested/fluid.c
 */

#include "../test_harness.h"
#include "tensor/tensor.h"
#include "nested/fluid.h"

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
 * Test: Backward Linear Gradient Computation
 * ============================================ */
void test_backward_linear_basic(void) {
    TEST_BEGIN("Backward Linear Basic");
    
    // Setup:
    // X: [1, 2] shape [1, 2] (1 sample, 2 features)
    // W: [2, 1] (2 inputs, 1 output)  [0.5, 0.5]
    // Y = X @ W = 1*0.5 + 2*0.5 = 1.5
    // Target = 2.0
    // dL/dY = Y - Target = 1.5 - 2.0 = -0.5
    // dL/dW = X^T @ dL/dY
    // dL/dW[0] = X[0,0] * dL/dY = 1.0 * -0.5 = -0.5
    // dL/dW[1] = X[0,1] * dL/dY = 2.0 * -0.5 = -1.0
    
    t_bf16 buf_x[2], buf_w[2], buf_grad_w[2], buf_grad_out[1];
    
    int shape_x[] = {1, 2};
    int shape_w[] = {2, 1};
    int shape_grad_out[] = {1, 1};
    
    float x_vals[] = {1.0f, 2.0f};
    float w_vals[] = {0.5f, 0.5f};
    float grad_out_vals[] = {-0.5f};
    
    t_tensor t_x = make_tensor(buf_x, shape_x, 2, x_vals);
    t_tensor t_w = make_tensor(buf_w, shape_w, 2, w_vals);
    t_tensor t_grad_w = make_tensor(buf_grad_w, shape_w, 2, NULL);
    t_tensor t_grad_out = make_tensor(buf_grad_out, shape_grad_out, 2, grad_out_vals);
    
    t_fluid_param fp = { &t_w, &t_grad_w };
    
    backward_linear(&fp, &t_x, &t_grad_out);
    
    // Expected gradients
    ASSERT_NEAR(-0.5f, bf16_to_float(((t_bf16*)t_grad_w.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(-1.0f, bf16_to_float(((t_bf16*)t_grad_w.data)[1]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: SGD Optimizer Step
 * ============================================ */
void test_optimizer_sgd_basic(void) {
    TEST_BEGIN("SGD Optimizer Basic");
    
    // W = [0.5, 0.5]
    // grad = [-0.5, -1.0]
    // lr = 0.1
    // W_new = W - lr * grad
    // W_new[0] = 0.5 - 0.1 * (-0.5) = 0.5 + 0.05 = 0.55
    // W_new[1] = 0.5 - 0.1 * (-1.0) = 0.5 + 0.10 = 0.60
    
    t_bf16 buf_w[2], buf_grad[2];
    int shape[] = {2, 1};
    
    float w_vals[] = {0.5f, 0.5f};
    float grad_vals[] = {-0.5f, -1.0f};
    
    t_tensor t_w = make_tensor(buf_w, shape, 2, w_vals);
    t_tensor t_grad = make_tensor(buf_grad, shape, 2, grad_vals);
    
    t_fluid_param fp = { &t_w, &t_grad };
    
    optimizer_sgd(&fp, 0.1f);
    
    ASSERT_NEAR(0.55f, bf16_to_float(((t_bf16*)t_w.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(0.60f, bf16_to_float(((t_bf16*)t_w.data)[1]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: Full Backward + SGD Pipeline
 * ============================================ */
void test_backward_sgd_pipeline(void) {
    TEST_BEGIN("Full Backward + SGD Pipeline");
    
    t_bf16 buf_x[2], buf_w[2], buf_grad_w[2], buf_grad_out[1];
    
    int shape_x[] = {1, 2};
    int shape_w[] = {2, 1};
    int shape_grad_out[] = {1, 1};
    
    float x_vals[] = {1.0f, 2.0f};
    float w_vals[] = {0.5f, 0.5f};
    float grad_out_vals[] = {-0.5f};  // (pred - target) when pred < target
    
    t_tensor t_x = make_tensor(buf_x, shape_x, 2, x_vals);
    t_tensor t_w = make_tensor(buf_w, shape_w, 2, w_vals);
    t_tensor t_grad_w = make_tensor(buf_grad_w, shape_w, 2, NULL);
    t_tensor t_grad_out = make_tensor(buf_grad_out, shape_grad_out, 2, grad_out_vals);
    
    t_fluid_param fp = { &t_w, &t_grad_w };
    
    // Step 1: Compute gradients
    backward_linear(&fp, &t_x, &t_grad_out);
    
    // Step 2: Apply SGD
    optimizer_sgd(&fp, 0.1f);
    
    // Verify final weights
    ASSERT_NEAR(0.55f, bf16_to_float(((t_bf16*)t_w.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(0.60f, bf16_to_float(((t_bf16*)t_w.data)[1]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: Zero Gradient
 * ============================================ */
void test_sgd_zero_gradient(void) {
    TEST_BEGIN("SGD with Zero Gradient");
    
    t_bf16 buf_w[2], buf_grad[2];
    int shape[] = {2};
    
    float w_vals[] = {1.0f, 2.0f};
    float grad_vals[] = {0.0f, 0.0f};
    
    t_tensor t_w = make_tensor(buf_w, shape, 1, w_vals);
    t_tensor t_grad = make_tensor(buf_grad, shape, 1, grad_vals);
    
    t_fluid_param fp = { &t_w, &t_grad };
    
    optimizer_sgd(&fp, 0.1f);
    
    // Weights should remain unchanged
    ASSERT_NEAR(1.0f, bf16_to_float(((t_bf16*)t_w.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(2.0f, bf16_to_float(((t_bf16*)t_w.data)[1]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: Large Learning Rate
 * ============================================ */
void test_sgd_large_lr(void) {
    TEST_BEGIN("SGD with Large Learning Rate");
    
    t_bf16 buf_w[2], buf_grad[2];
    int shape[] = {2};
    
    float w_vals[] = {1.0f, 1.0f};
    float grad_vals[] = {1.0f, -1.0f};
    
    t_tensor t_w = make_tensor(buf_w, shape, 1, w_vals);
    t_tensor t_grad = make_tensor(buf_grad, shape, 1, grad_vals);
    
    t_fluid_param fp = { &t_w, &t_grad };
    
    optimizer_sgd(&fp, 1.0f);  // lr = 1.0
    
    // W = W - 1.0 * grad
    ASSERT_NEAR(0.0f, bf16_to_float(((t_bf16*)t_w.data)[0]), BF16_TOLERANCE);  // 1 - 1 = 0
    ASSERT_NEAR(2.0f, bf16_to_float(((t_bf16*)t_w.data)[1]), BF16_TOLERANCE);  // 1 - (-1) = 2
    
    TEST_END();
}

/* ============================================
 * Test: Multiple SGD Steps (Convergence)
 * ============================================ */
void test_sgd_convergence(void) {
    TEST_BEGIN("SGD Convergence (Multiple Steps)");
    
    // Simple 1D regression: y = w * x, target = 2
    // x = 1.0, so y = w
    // Loss = 0.5 * (w - 2)^2
    // dL/dw = w - 2
    // Starting w = 0.0, should converge to 2.0
    
    t_bf16 buf_w[1], buf_grad[1];
    int shape[] = {1};
    
    float w_val = 0.0f;
    float target = 2.0f;
    float lr = 0.2f;
    
    t_tensor t_w = make_tensor(buf_w, shape, 1, &w_val);
    t_tensor t_grad = tensor_view(buf_grad, shape, 1);
    
    t_fluid_param fp = { &t_w, &t_grad };
    
    for (int step = 0; step < 20; step++) {
        float w = bf16_to_float(((t_bf16*)t_w.data)[0]);
        float grad = w - target;  // dL/dw
        ((t_bf16*)t_grad.data)[0] = float_to_bf16(grad);
        
        optimizer_sgd(&fp, lr);
    }
    
    float final_w = bf16_to_float(((t_bf16*)t_w.data)[0]);
    LOG_INFO("Final w after 20 steps: %.4f (target: 2.0)", final_w);
    
    // Should be close to 2.0
    ASSERT_NEAR(2.0f, final_w, 0.1f);
    
    TEST_END();
}

/* ============================================
 * Test: Backward Linear Larger Matrix
 * ============================================ */
void test_backward_linear_larger(void) {
    TEST_BEGIN("Backward Linear [2x3] x [3x2]");
    
    // X: [2, 3] (2 samples, 3 features)
    // W: [3, 2] (3 inputs, 2 outputs)
    // grad_out: [2, 2]
    // grad_W: [3, 2] = X^T @ grad_out
    
    t_bf16 buf_x[6], buf_w[6], buf_grad_w[6], buf_grad_out[4];
    
    int shape_x[] = {2, 3};
    int shape_w[] = {3, 2};
    int shape_grad_out[] = {2, 2};
    
    // X = [[1, 2, 3], [4, 5, 6]]
    float x_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float w_vals[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};  // Dummy
    // grad_out = [[1, 0], [0, 1]]
    float grad_out_vals[] = {1.0f, 0.0f, 0.0f, 1.0f};
    
    t_tensor t_x = make_tensor(buf_x, shape_x, 2, x_vals);
    t_tensor t_w = make_tensor(buf_w, shape_w, 2, w_vals);
    t_tensor t_grad_w = make_tensor(buf_grad_w, shape_w, 2, NULL);
    t_tensor t_grad_out = make_tensor(buf_grad_out, shape_grad_out, 2, grad_out_vals);
    
    t_fluid_param fp = { &t_w, &t_grad_w };
    
    backward_linear(&fp, &t_x, &t_grad_out);
    
    // grad_W = X^T @ grad_out
    // X^T = [[1, 4], [2, 5], [3, 6]]  (3x2)
    // grad_out = [[1, 0], [0, 1]]  (2x2)
    // grad_W[0,0] = 1*1 + 4*0 = 1
    // grad_W[0,1] = 1*0 + 4*1 = 4
    // grad_W[1,0] = 2*1 + 5*0 = 2
    // grad_W[1,1] = 2*0 + 5*1 = 5
    // grad_W[2,0] = 3*1 + 6*0 = 3
    // grad_W[2,1] = 3*0 + 6*1 = 6
    
    ASSERT_NEAR(1.0f, bf16_to_float(((t_bf16*)t_grad_w.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(4.0f, bf16_to_float(((t_bf16*)t_grad_w.data)[1]), BF16_TOLERANCE);
    ASSERT_NEAR(2.0f, bf16_to_float(((t_bf16*)t_grad_w.data)[2]), BF16_TOLERANCE);
    ASSERT_NEAR(5.0f, bf16_to_float(((t_bf16*)t_grad_w.data)[3]), BF16_TOLERANCE);
    ASSERT_NEAR(3.0f, bf16_to_float(((t_bf16*)t_grad_w.data)[4]), BF16_TOLERANCE);
    ASSERT_NEAR(6.0f, bf16_to_float(((t_bf16*)t_grad_w.data)[5]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Main
 * ============================================ */
int main(void) {
    TEST_SUITE_BEGIN("Nested Learning (Fluid Weights)");
    
    test_backward_linear_basic();
    test_optimizer_sgd_basic();
    test_backward_sgd_pipeline();
    test_sgd_zero_gradient();
    test_sgd_large_lr();
    test_sgd_convergence();
    test_backward_linear_larger();
    
    TEST_SUITE_END();
}
