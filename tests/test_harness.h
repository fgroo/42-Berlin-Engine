/*
 * 42-BERLIN-ENGINE Test Harness
 * ===========================================
 * Lightweight testing framework for pure C.
 * No dependencies. Just assertions and colors.
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================
 * ANSI Color Codes
 * ============================================ */
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_YELLOW  "\x1b[33m"
#define COLOR_BLUE    "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN    "\x1b[36m"
#define COLOR_RESET   "\x1b[0m"
#define COLOR_BOLD    "\x1b[1m"

/* ============================================
 * Test Statistics (Global)
 * ============================================ */
static int _tests_run = 0;
static int _tests_passed = 0;
static int _tests_failed = 0;
static int _current_test_failed = 0;
static const char *_current_test_name = NULL;

/* ============================================
 * Test Macros
 * ============================================ */

#define TEST_SUITE_BEGIN(name) \
    do { \
        printf("\n" COLOR_BOLD COLOR_CYAN "═══════════════════════════════════════════\n"); \
        printf(" TEST SUITE: %s\n", name); \
        printf("═══════════════════════════════════════════" COLOR_RESET "\n\n"); \
    } while(0)

#define TEST_SUITE_END() \
    do { \
        printf("\n" COLOR_BOLD "───────────────────────────────────────────\n"); \
        printf(" RESULTS: "); \
        if (_tests_failed == 0) { \
            printf(COLOR_GREEN "ALL %d TESTS PASSED ✓" COLOR_RESET "\n", _tests_passed); \
        } else { \
            printf(COLOR_RED "%d FAILED" COLOR_RESET ", " COLOR_GREEN "%d PASSED" COLOR_RESET " (of %d)\n", \
                   _tests_failed, _tests_passed, _tests_run); \
        } \
        printf(COLOR_BOLD "───────────────────────────────────────────" COLOR_RESET "\n\n"); \
        return _tests_failed; \
    } while(0)

#define TEST_BEGIN(name) \
    do { \
        _tests_run++; \
        _current_test_failed = 0; \
        _current_test_name = name; \
        printf(COLOR_YELLOW "▶ " COLOR_RESET "%s ... ", name); \
        fflush(stdout); \
    } while(0)

#define TEST_END() \
    do { \
        if (_current_test_failed) { \
            _tests_failed++; \
            printf(COLOR_RED "[FAIL]" COLOR_RESET "\n"); \
        } else { \
            _tests_passed++; \
            printf(COLOR_GREEN "[PASS]" COLOR_RESET "\n"); \
        } \
    } while(0)

/* ============================================
 * Assertion Macros
 * ============================================ */

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_TRUE failed: " #cond COLOR_RESET "\n"); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_FALSE(cond) \
    do { \
        if (cond) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_FALSE failed: " #cond COLOR_RESET "\n"); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_EQ(expected, actual) \
    do { \
        if ((expected) != (actual)) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_EQ failed:" COLOR_RESET "\n"); \
            printf("      Expected: %lld\n", (long long)(expected)); \
            printf("      Actual:   %lld\n", (long long)(actual)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_EQ_PTR(expected, actual) \
    do { \
        if ((expected) != (actual)) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_EQ_PTR failed:" COLOR_RESET "\n"); \
            printf("      Expected: %p\n", (void*)(expected)); \
            printf("      Actual:   %p\n", (void*)(actual)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_NEQ(not_expected, actual) \
    do { \
        if ((not_expected) == (actual)) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_NEQ failed:" COLOR_RESET "\n"); \
            printf("      Value should not equal: %lld\n", (long long)(not_expected)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_NOT_NULL(ptr) \
    do { \
        if ((ptr) == NULL) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_NOT_NULL failed: " #ptr " is NULL" COLOR_RESET "\n"); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_NULL(ptr) \
    do { \
        if ((ptr) != NULL) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_NULL failed: " #ptr " = %p" COLOR_RESET "\n", (void*)(ptr)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_NEAR(expected, actual, tolerance) \
    do { \
        double _e = (double)(expected); \
        double _a = (double)(actual); \
        double _t = (double)(tolerance); \
        if (fabs(_e - _a) > _t) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_NEAR failed:" COLOR_RESET "\n"); \
            printf("      Expected: %.9g\n", _e); \
            printf("      Actual:   %.9g\n", _a); \
            printf("      Delta:    %.9g (tolerance: %.9g)\n", fabs(_e - _a), _t); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_STR_EQ(expected, actual) \
    do { \
        if (strcmp((expected), (actual)) != 0) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_STR_EQ failed:" COLOR_RESET "\n"); \
            printf("      Expected: \"%s\"\n", (expected)); \
            printf("      Actual:   \"%s\"\n", (actual)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_MEM_EQ(expected, actual, size) \
    do { \
        if (memcmp((expected), (actual), (size)) != 0) { \
            if (!_current_test_failed) printf("\n"); \
            printf("    " COLOR_RED "✗ ASSERT_MEM_EQ failed (size=%zu)" COLOR_RESET "\n", (size_t)(size)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            _current_test_failed = 1; \
        } \
    } while(0)

/* ============================================
 * Utility Macros
 * ============================================ */

#define SKIP_TEST(reason) \
    do { \
        printf(COLOR_YELLOW "[SKIP: %s]" COLOR_RESET "\n", reason); \
        return; \
    } while(0)

#define LOG_INFO(fmt, ...) \
    printf("    " COLOR_BLUE "ℹ " COLOR_RESET fmt "\n", ##__VA_ARGS__)

#define LOG_DEBUG(fmt, ...) \
    printf("    " COLOR_MAGENTA "⚙ " COLOR_RESET fmt "\n", ##__VA_ARGS__)

/* ============================================
 * BF16 Tolerance (for floating point tests)
 * BF16 has ~3 significant digits of precision
 * ============================================ */
#define BF16_TOLERANCE 0.05f
#define BF16_TOLERANCE_LOOSE 0.2f

/* ============================================
 * Hex dump utility (for debugging)
 * ============================================ */
static inline void hexdump(const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i++) {
        printf("%02x ", bytes[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    if (size % 16 != 0) printf("\n");
}

#endif /* TEST_HARNESS_H */
