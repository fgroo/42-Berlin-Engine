/*
 * 42-BERLIN-ENGINE: t_tensor & BF16 Tests
 * ======================================
 * Tests for tensor/tensor.c
 */

#include "../test_harness.h"
#include "tensor/tensor.h"
#include <float.h>

/* ============================================
 * Test: BF16 Conversion Round-Trip
 * ============================================ */
void test_bf16_roundtrip(void) {
    TEST_BEGIN("BF16 Round-Trip Conversion");
    
    float test_values[] = {
        0.0f, 1.0f, -1.0f, 3.14159f, -3.14159f,
        100.0f, -100.0f, 0.001f, -0.001f,
        65504.0f,  // Max half precision value (sanity)
    };
    int n = sizeof(test_values) / sizeof(test_values[0]);
    
    for (int i = 0; i < n; i++) {
        float original = test_values[i];
        t_bf16 bf = float_to_bf16(original);
        float recovered = bf16_to_float(bf);
        
        // BF16 has ~3 significant digits, tolerance ~1%
        if (original != 0.0f) {
            ASSERT_NEAR(original, recovered, fabsf(original) * 0.01f);
        } else {
            ASSERT_EQ(0.0f, recovered);
        }
    }
    
    TEST_END();
}

/* ============================================
 * Test: BF16 Zero Values
 * ============================================ */
void test_bf16_zero(void) {
    TEST_BEGIN("BF16 Zero Values");
    
    // Positive zero
    t_bf16 zero = float_to_bf16(0.0f);
    float fzero = bf16_to_float(zero);
    ASSERT_EQ(0.0f, fzero);
    
    // Negative zero
    t_bf16 neg_zero = float_to_bf16(-0.0f);
    float fneg_zero = bf16_to_float(neg_zero);
    ASSERT_TRUE(fneg_zero == 0.0f || fneg_zero == -0.0f);
    
    TEST_END();
}

/* ============================================
 * Test: BF16 Infinity
 * ============================================ */
void test_bf16_infinity(void) {
    TEST_BEGIN("BF16 Infinity Handling");
    
    float pos_inf = INFINITY;
    float neg_inf = -INFINITY;
    
    t_bf16 bf_pos_inf = float_to_bf16(pos_inf);
    t_bf16 bf_neg_inf = float_to_bf16(neg_inf);
    
    float recovered_pos = bf16_to_float(bf_pos_inf);
    float recovered_neg = bf16_to_float(bf_neg_inf);
    
    ASSERT_TRUE(isinf(recovered_pos) && recovered_pos > 0);
    ASSERT_TRUE(isinf(recovered_neg) && recovered_neg < 0);
    
    TEST_END();
}

/* ============================================
 * Test: BF16 NaN
 * ============================================ */
void test_bf16_nan(void) {
    TEST_BEGIN("BF16 NaN Handling");
    
    float nan_val = NAN;
    t_bf16 bf_nan = float_to_bf16(nan_val);
    float recovered = bf16_to_float(bf_nan);
    
    ASSERT_TRUE(isnan(recovered));
    
    TEST_END();
}

/* ============================================
 * Test: BF16 Precision Loss
 * ============================================ */
void test_bf16_precision(void) {
    TEST_BEGIN("BF16 Precision (3 significant digits)");
    
    // BF16 has 7 bits of mantissa = ~2.4 decimal digits
    // Values that differ only in low bits should collapse
    
    float a = 1.000f;
    float b = 1.001f;  // ~0.1% difference
    float c = 1.01f;   // ~1% difference
    
    t_bf16 bf_a = float_to_bf16(a);
    t_bf16 bf_b = float_to_bf16(b);
    t_bf16 bf_c = float_to_bf16(c);
    
    // a and b might round to same BF16 (that's expected)
    // a and c should definitely be different
    ASSERT_TRUE(bf_a != bf_c);
    
    LOG_INFO("a=%.6f -> bf16=0x%04x", a, bf_a);
    LOG_INFO("b=%.6f -> bf16=0x%04x", b, bf_b);
    LOG_INFO("c=%.6f -> bf16=0x%04x", c, bf_c);
    
    TEST_END();
}

/* ============================================
 * Test: t_tensor View 1D
 * ============================================ */
void test_tensor_view_1d(void) {
    TEST_BEGIN("t_tensor View 1D");
    
    t_bf16 data[4] = {0};
    int shape[] = {4};
    
    t_tensor t = tensor_view(data, shape, 1);
    
    ASSERT_EQ(4, t.shape[0]);
    ASSERT_EQ(1, t.ndim);
    ASSERT_EQ(4, t.size);
    ASSERT_EQ(1, t.stride[0]);
    ASSERT_EQ_PTR(data, t.data);
    
    TEST_END();
}

/* ============================================
 * Test: t_tensor View 2D
 * ============================================ */
void test_tensor_view_2d(void) {
    TEST_BEGIN("t_tensor View 2D");
    
    t_bf16 data[6] = {0};
    int shape[] = {2, 3};
    
    t_tensor t = tensor_view(data, shape, 2);
    
    ASSERT_EQ(2, t.shape[0]);
    ASSERT_EQ(3, t.shape[1]);
    ASSERT_EQ(2, t.ndim);
    ASSERT_EQ(6, t.size);
    
    // Row-major strides: stride[0] = 3, stride[1] = 1
    ASSERT_EQ(3, t.stride[0]);
    ASSERT_EQ(1, t.stride[1]);
    
    TEST_END();
}

/* ============================================
 * Test: t_tensor View 3D
 * ============================================ */
void test_tensor_view_3d(void) {
    TEST_BEGIN("t_tensor View 3D");
    
    t_bf16 data[24] = {0};
    int shape[] = {2, 3, 4};
    
    t_tensor t = tensor_view(data, shape, 3);
    
    ASSERT_EQ(2, t.shape[0]);
    ASSERT_EQ(3, t.shape[1]);
    ASSERT_EQ(4, t.shape[2]);
    ASSERT_EQ(3, t.ndim);
    ASSERT_EQ(24, t.size);
    
    // Row-major strides: stride = [12, 4, 1]
    ASSERT_EQ(12, t.stride[0]);
    ASSERT_EQ(4, t.stride[1]);
    ASSERT_EQ(1, t.stride[2]);
    
    TEST_END();
}

/* ============================================
 * Test: t_tensor View 4D
 * ============================================ */
void test_tensor_view_4d(void) {
    TEST_BEGIN("t_tensor View 4D (Batch, Head, Seq, Dim)");
    
    t_bf16 data[480] = {0};  // 2 * 4 * 6 * 10 = 480
    int shape[] = {2, 4, 6, 10};
    
    t_tensor t = tensor_view(data, shape, 4);
    
    ASSERT_EQ(2, t.shape[0]);
    ASSERT_EQ(4, t.shape[1]);
    ASSERT_EQ(6, t.shape[2]);
    ASSERT_EQ(10, t.shape[3]);
    ASSERT_EQ(4, t.ndim);
    ASSERT_EQ(480, t.size);
    
    // Row-major strides: [240, 60, 10, 1]
    ASSERT_EQ(240, t.stride[0]);  // 4 * 6 * 10
    ASSERT_EQ(60, t.stride[1]);   // 6 * 10
    ASSERT_EQ(10, t.stride[2]);   // 10
    ASSERT_EQ(1, t.stride[3]);
    
    TEST_END();
}

/* ============================================
 * Test: t_tensor Data Access
 * ============================================ */
void test_tensor_data_access(void) {
    TEST_BEGIN("t_tensor Data Read/Write");
    
    t_bf16 data[4];
    int shape[] = {2, 2};
    
    t_tensor t = tensor_view(data, shape, 2);
    
    // Write values
    ((t_bf16*)t.data)[0] = float_to_bf16(1.0f);
    ((t_bf16*)t.data)[1] = float_to_bf16(2.0f);
    ((t_bf16*)t.data)[2] = float_to_bf16(3.0f);
    ((t_bf16*)t.data)[3] = float_to_bf16(4.0f);
    
    // Read back
    ASSERT_NEAR(1.0f, bf16_to_float(((t_bf16*)t.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(2.0f, bf16_to_float(((t_bf16*)t.data)[1]), BF16_TOLERANCE);
    ASSERT_NEAR(3.0f, bf16_to_float(((t_bf16*)t.data)[2]), BF16_TOLERANCE);
    ASSERT_NEAR(4.0f, bf16_to_float(((t_bf16*)t.data)[3]), BF16_TOLERANCE);
    
    TEST_END();
}

/* ============================================
 * Test: BF16 Specific Values
 * ============================================ */
void test_bf16_specific_values(void) {
    TEST_BEGIN("BF16 Specific Known Values");
    
    // Known BF16 encodings (from IEEE 754)
    // 1.0f in BF16: 0x3F80
    t_bf16 one = float_to_bf16(1.0f);
    ASSERT_EQ(0x3F80, one);
    
    // 2.0f in BF16: 0x4000
    t_bf16 two = float_to_bf16(2.0f);
    ASSERT_EQ(0x4000, two);
    
    // -1.0f in BF16: 0xBF80
    t_bf16 neg_one = float_to_bf16(-1.0f);
    ASSERT_EQ(0xBF80, neg_one);
    
    // 0.5f in BF16: 0x3F00
    t_bf16 half = float_to_bf16(0.5f);
    ASSERT_EQ(0x3F00, half);
    
    TEST_END();
}

/* ============================================
 * Test: Large t_tensor
 * ============================================ */
void test_tensor_large(void) {
    TEST_BEGIN("Large t_tensor (4096 elements)");
    
    t_bf16 *data = (t_bf16*)malloc(4096 * sizeof(t_bf16));
    ASSERT_NOT_NULL(data);
    
    int shape[] = {64, 64};
    t_tensor t = tensor_view(data, shape, 2);
    
    ASSERT_EQ(4096, t.size);
    
    // Fill and verify
    for (int i = 0; i < 4096; i++) {
        ((t_bf16*)t.data)[i] = float_to_bf16((float)i);
    }
    
    // Spot check
    ASSERT_NEAR(0.0f, bf16_to_float(((t_bf16*)t.data)[0]), BF16_TOLERANCE);
    ASSERT_NEAR(100.0f, bf16_to_float(((t_bf16*)t.data)[100]), BF16_TOLERANCE);
    ASSERT_NEAR(1000.0f, bf16_to_float(((t_bf16*)t.data)[1000]), BF16_TOLERANCE_LOOSE);
    
    free(data);
    TEST_END();
}

/* ============================================
 * Main
 * ============================================ */
int main(void) {
    TEST_SUITE_BEGIN("t_tensor & BF16");
    
    test_bf16_roundtrip();
    test_bf16_zero();
    test_bf16_infinity();
    test_bf16_nan();
    test_bf16_precision();
    test_bf16_specific_values();
    
    test_tensor_view_1d();
    test_tensor_view_2d();
    test_tensor_view_3d();
    test_tensor_view_4d();
    test_tensor_data_access();
    test_tensor_large();
    
    TEST_SUITE_END();
}
