/*
 * 42-BERLIN-ENGINE: t_arena Allocator Tests
 * ========================================
 * Tests for memory/arena.c
 */

#include "../test_harness.h"
#include "memory/arena.h"

/* ============================================
 * Test: Basic Initialization
 * ============================================ */
void test_arena_init(void) {
    TEST_BEGIN("t_arena Initialization");
    
    t_arena arena;
    arena_init(&arena, 1024);
    
    ASSERT_NOT_NULL(arena.base);
    ASSERT_EQ(1024, arena.size);
    ASSERT_EQ(0, arena.offset);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Simple Allocation
 * ============================================ */
void test_arena_simple_alloc(void) {
    TEST_BEGIN("Simple Allocation");
    
    t_arena arena;
    arena_init(&arena, 4096);
    
    void *ptr = arena_alloc_or_die(&arena, 100);
    ASSERT_NOT_NULL(ptr);
    ASSERT_TRUE(arena.offset >= 100); // May include alignment padding
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: 64-Byte Alignment
 * ============================================ */
void test_arena_alignment(void) {
    TEST_BEGIN("64-Byte Alignment");
    
    t_arena arena;
    arena_init(&arena, 4096);
    
    // First allocation
    void *ptr1 = arena_alloc_or_die(&arena, 50);
    ASSERT_NOT_NULL(ptr1);
    ASSERT_EQ(0, (uintptr_t)ptr1 % 64); // Should be 64-byte aligned
    
    // Second allocation should also be aligned
    void *ptr2 = arena_alloc_or_die(&arena, 25);
    ASSERT_NOT_NULL(ptr2);
    ASSERT_EQ(0, (uintptr_t)ptr2 % 64);
    
    // Verify they don't overlap
    ASSERT_TRUE((uintptr_t)ptr2 >= (uintptr_t)ptr1 + 50);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Multiple Allocations
 * ============================================ */
void test_arena_multiple_allocs(void) {
    TEST_BEGIN("Multiple Allocations");
    
    t_arena arena;
    arena_init(&arena, 8192);
    
    void *ptrs[10];
    for (int i = 0; i < 10; i++) {
        ptrs[i] = arena_alloc_or_die(&arena, 100);
        ASSERT_NOT_NULL(ptrs[i]);
    }
    
    // Verify all pointers are unique and don't overlap
    for (int i = 0; i < 10; i++) {
        for (int j = i + 1; j < 10; j++) {
            ASSERT_TRUE(ptrs[i] != ptrs[j]);
        }
    }
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Reset Functionality
 * ============================================ */
void test_arena_reset(void) {
    TEST_BEGIN("Reset Functionality");
    
    t_arena arena;
    arena_init(&arena, 4096);
    
    void *ptr1 = arena_alloc_or_die(&arena, 1000);
    ASSERT_NOT_NULL(ptr1);
    size_t offset_before = arena.offset;
    ASSERT_TRUE(offset_before >= 1000);
    
    arena_reset(&arena);
    ASSERT_EQ(0, arena.offset);
    
    // Can allocate again from start
    void *ptr2 = arena_alloc_or_die(&arena, 500);
    ASSERT_NOT_NULL(ptr2);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Zero-Fill Guarantee
 * ============================================ */
void test_arena_zero_fill(void) {
    TEST_BEGIN("Zero-Fill Guarantee");
    
    t_arena arena;
    arena_init(&arena, 4096);
    
    uint8_t *ptr = (uint8_t*)arena_alloc_or_die(&arena, 256);
    ASSERT_NOT_NULL(ptr);
    
    // t_arena should zero-fill memory (per implementation in arena.c)
    int all_zero = 1;
    for (int i = 0; i < 256; i++) {
        if (ptr[i] != 0) {
            all_zero = 0;
            break;
        }
    }
    ASSERT_TRUE(all_zero);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Free Clears State
 * ============================================ */
void test_arena_free(void) {
    TEST_BEGIN("Free Clears State");
    
    t_arena arena;
    arena_init(&arena, 4096);
    
    arena_alloc_or_die(&arena, 100);
    arena_free(&arena);
    
    ASSERT_NULL(arena.base);
    ASSERT_EQ(0, arena.size);
    ASSERT_EQ(0, arena.offset);
    
    TEST_END();
}

/* ============================================
 * Test: Large Allocation
 * ============================================ */
void test_arena_large_alloc(void) {
    TEST_BEGIN("Large Allocation (1MB)");
    
    t_arena arena;
    size_t size = 2 * 1024 * 1024; // 2MB
    arena_init(&arena, size);
    
    void *ptr = arena_alloc_or_die(&arena, 1024 * 1024); // 1MB
    ASSERT_NOT_NULL(ptr);
    
    // Can still allocate more
    void *ptr2 = arena_alloc_or_die(&arena, 512 * 1024); // 512KB
    ASSERT_NOT_NULL(ptr2);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Contiguous Allocations
 * ============================================ */
void test_arena_contiguous(void) {
    TEST_BEGIN("Contiguous Allocations (Cache-Friendly)");
    
    t_arena arena;
    arena_init(&arena, 4096);
    
    // Allocate a series of equal-sized chunks
    void *ptr1 = arena_alloc_or_die(&arena, 64);
    void *ptr2 = arena_alloc_or_die(&arena, 64);
    void *ptr3 = arena_alloc_or_die(&arena, 64);
    
    ASSERT_NOT_NULL(ptr1);
    ASSERT_NOT_NULL(ptr2);
    ASSERT_NOT_NULL(ptr3);
    
    // Due to 64-byte alignment and 64-byte size, they should be exactly 64 bytes apart
    ASSERT_EQ(64, (uintptr_t)ptr2 - (uintptr_t)ptr1);
    ASSERT_EQ(64, (uintptr_t)ptr3 - (uintptr_t)ptr2);
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Test: Stress Test - Many Small Allocations
 * ============================================ */
void test_arena_stress_small(void) {
    TEST_BEGIN("Stress: 1000 Small Allocations");
    
    t_arena arena;
    arena_init(&arena, 1024 * 1024); // 1MB
    
    for (int i = 0; i < 1000; i++) {
        void *ptr = arena_alloc_or_die(&arena, 8);
        ASSERT_NOT_NULL(ptr);
    }
    
    arena_free(&arena);
    TEST_END();
}

/* ============================================
 * Main
 * ============================================ */
int main(void) {
    TEST_SUITE_BEGIN("t_arena Allocator");
    
    test_arena_init();
    test_arena_simple_alloc();
    test_arena_alignment();
    test_arena_multiple_allocs();
    test_arena_reset();
    test_arena_zero_fill();
    test_arena_free();
    test_arena_large_alloc();
    test_arena_contiguous();
    test_arena_stress_small();
    
    TEST_SUITE_END();
}
