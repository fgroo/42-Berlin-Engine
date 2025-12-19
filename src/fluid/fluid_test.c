/* Quick test to create a v2 fluid file */
#include "../fluid/fluid_io.h"
#include <stdio.h>

int main(void) {
    t_fluid_header h = fluid_create_header("test", "unit-test", 0x123456789ABCDEF0ULL);
    fluid_set_description(&h, "Test file for Fluid Protocol v2 verification");
    h.n_entries = 3;
    h.flags = FLUID_FLAG_VERIFIED;
    
    t_fluid_entry entries[3] = {
        { .context_hash = 0xAAAA, .target_token = 100, .weight = 0.5f },
        { .context_hash = 0xBBBB, .target_token = 200, .weight = 0.75f },
        { .context_hash = 0xCCCC, .target_token = 300, .weight = 1.0f }
    };
    
    if (fluid_write_file("test.fluid", &h, entries) == FLUID_OK)
        printf("Created test.fluid successfully!\n");
    else
        printf("Failed to create test.fluid\n");
    return 0;
}
