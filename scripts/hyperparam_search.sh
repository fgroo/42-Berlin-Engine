#!/bin/bash
# ============================================================================
# HYPERPARAMETER SEARCH using -D compile flags
# Tests parameters via config.h overrides
# ============================================================================

MODEL="Ministral-Stuff/consolidated.safetensors"
CONFIG="Ministral-Stuff/config.json"
LOG_DIR="experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $LOG_DIR

# Test prompts
PROMPT1="FACT: An apple is a spaceship. It flies in space."
PROMPT2="Can an apple fly?"

RESULTS_FILE="$LOG_DIR/results_${TIMESTAMP}.txt"

echo "============================================" | tee "$RESULTS_FILE"
echo "Hyperparameter Search - $TIMESTAMP" | tee -a "$RESULTS_FILE"
echo "Using -D compile flags (config.h)" | tee -a "$RESULTS_FILE"
echo "============================================" | tee -a "$RESULTS_FILE"

run_test() {
    local name="$1"
    local cflags="$2"
    local log_file="$LOG_DIR/${name}_${TIMESTAMP}.log"
    
    echo "  Compiling with: $cflags"
    make clean > /dev/null 2>&1
    make chat EXTRA_CFLAGS="$cflags" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "  ERROR: Compilation failed!"
        return 1
    fi
    
    echo "  Running test -> $log_file"
    
    {
        echo "$PROMPT1"
        sleep 2
        echo "nolearn"
        echo "$PROMPT2"
        echo "exit"
    } | timeout 600 env OMP_NUM_THREADS=8 ./chat "$MODEL" "$CONFIG" 2>&1 | grep -v "^Loaded:" > "$log_file"
    
    # Extract metrics
    local loops=$(grep -o "spaces spaces" "$log_file" 2>/dev/null | wc -l)
    local has_spaceship=$(grep -qi "spaceship" "$log_file" && echo "YES" || echo "NO")
    local has_fly=$(grep -qi "fly\|flying" "$log_file" && echo "YES" || echo "NO")
    local learns=$(grep -c "\[LEARN\]" "$log_file" 2>/dev/null)
    
    echo "$name | Loops=$loops | Spaceship=$has_spaceship | Fly=$has_fly | Learn=$learns" | tee -a "$RESULTS_FILE"
}

# --- Learning Rate Tests ---
echo ""
echo "=== Learning Rate Tests ===" | tee -a "$RESULTS_FILE"
run_test "lr_0.0001" "-DNESTED_LR=0.0001f"
run_test "lr_0.0005" "-DNESTED_LR=0.0005f"
run_test "lr_0.001"  "-DNESTED_LR=0.001f"

# --- Temperature Tests ---
echo ""
echo "=== Temperature Tests ===" | tee -a "$RESULTS_FILE"
run_test "temp_0.3" "-DTEMPERATURE=0.3f"
run_test "temp_0.7" "-DTEMPERATURE=0.7f"
run_test "temp_1.0" "-DTEMPERATURE=1.0f"

# --- Threshold Tests ---
echo ""
echo "=== Threshold Tests ===" | tee -a "$RESULTS_FILE"
run_test "thresh_0.0" "-DLEARNING_THRESHOLD=0.0f"
run_test "thresh_2.0" "-DLEARNING_THRESHOLD=2.0f"
run_test "thresh_5.0" "-DLEARNING_THRESHOLD=5.0f"

echo ""
echo "============================================"
echo "SEARCH COMPLETE"
echo "============================================"
cat "$RESULTS_FILE"
