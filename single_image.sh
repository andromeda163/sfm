#!/bin/bash
# Single Image SIFT Comparison Script
# Compares C++ standalone, sift-wgpu, sift-wgpu wrapper, and sift-rust

# Default values
IMAGE="${1:-data/bear01/bear01_0001.jpg}"
OUTPUT_DIR="output/comparison"
MAX_FEATURES="${2:-10000}"
FIRST_OCTAVE="${3:--1}"
PEAK_THRESH="${4:-0.00667}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clear results file
RESULTS_FILE="$OUTPUT_DIR/results.txt"
> "$RESULTS_FILE"

echo "=============================================="
echo "SIFT Implementation Comparison"
echo "=============================================="
echo ""
echo "Image: $IMAGE"
echo "Max Features: $MAX_FEATURES"
echo "First Octave: $FIRST_OCTAVE"
echo "Peak Threshold: $PEAK_THRESH"
echo ""

# Function to calculate elapsed time
calc_time() {
    local start=$1
    local end=$2
    echo "$end $start" | awk '{printf "%.2f", $1 - $2}'
}

# 1. C++ Standalone (VLFeat)
if [ -x "./sift/build/sift_extract" ]; then
    echo -e "${YELLOW}Running C++ (VLFeat)...${NC}"
    mkdir -p "$OUTPUT_DIR/cpp"
    START=$(date +%s.%N)
    ./sift/build/sift_extract --first_octave $FIRST_OCTAVE --peak_thresh $PEAK_THRESH --max_features $MAX_FEATURES "$IMAGE" "$OUTPUT_DIR/cpp" > /tmp/sift_cpp.log 2>&1
    END=$(date +%s.%N)
    TIME=$(calc_time "$START" "$END")
    FEATURES=$(grep "Extracted" /tmp/sift_cpp.log | grep -oE '[0-9]+' | head -1)
    [ -z "$FEATURES" ] && FEATURES="N/A"
    echo "C++ (VLFeat)|$FEATURES|$TIME" >> "$RESULTS_FILE"
else
    echo -e "${RED}C++ implementation not found at ./sift/build/sift_extract${NC}"
    echo "C++ (VLFeat)|N/A|N/A" >> "$RESULTS_FILE"
fi

# 2. sift-wgpu (original benchmark)
if [ -x "./benchmark/target/release/sift-benchmark" ]; then
    echo -e "${YELLOW}Running sift-wgpu (original)...${NC}"
    START=$(date +%s.%N)
    ./benchmark/target/release/sift-benchmark --input "$IMAGE" --wgpu-only --max-features $MAX_FEATURES > /tmp/sift_wgpu.log 2>&1
    END=$(date +%s.%N)
    TIME=$(calc_time "$START" "$END")
    FEATURES=$(grep "features" /tmp/sift_wgpu.log | grep -oE '[0-9]+ features' | head -1 | grep -oE '[0-9]+')
    [ -z "$FEATURES" ] && FEATURES=$(grep "Avg Features" /tmp/sift_wgpu.log | awk '{print int($4)}')
    [ -z "$FEATURES" ] && FEATURES="N/A"
    echo "sift-wgpu|$FEATURES|$TIME" >> "$RESULTS_FILE"
else
    echo -e "${RED}sift-wgpu benchmark not found${NC}"
    echo "sift-wgpu|N/A|N/A" >> "$RESULTS_FILE"
fi

# 3. sift-wgpu wrapper
if [ -x "./sift-wgpu/target/release/sift_extract_wgpu" ]; then
    echo -e "${YELLOW}Running sift-wgpu wrapper...${NC}"
    mkdir -p "$OUTPUT_DIR/sift_wgpu_wrapper"
    START=$(date +%s.%N)
    ./sift-wgpu/target/release/sift_extract_wgpu --first-octave=$FIRST_OCTAVE --peak-thresh=$PEAK_THRESH --max-features=$MAX_FEATURES --max-orientations=2 "$IMAGE" "$OUTPUT_DIR/sift_wgpu_wrapper" > /tmp/sift_wgpu_wrapper.log 2>&1
    END=$(date +%s.%N)
    TIME=$(calc_time "$START" "$END")
    FEATURES=$(grep "Extracted" /tmp/sift_wgpu_wrapper.log | grep -oE '[0-9]+' | head -1)
    [ -z "$FEATURES" ] && FEATURES=$(grep "Features:" /tmp/sift_wgpu_wrapper.log | grep -oE '[0-9]+' | tail -1)
    [ -z "$FEATURES" ] && FEATURES="N/A"
    echo "sift-wgpu wrapper|$FEATURES|$TIME" >> "$RESULTS_FILE"
else
    echo -e "${RED}sift-wgpu wrapper not found${NC}"
    echo "sift-wgpu wrapper|N/A|N/A" >> "$RESULTS_FILE"
fi

# 4. Rust from scratch
if [ -x "./sift-rust/target/release/sift_extract" ]; then
    echo -e "${YELLOW}Running Rust (from scratch)...${NC}"
    mkdir -p "$OUTPUT_DIR/sift_rust"
    START=$(date +%s.%N)
    ./sift-rust/target/release/sift_extract --first-octave=$FIRST_OCTAVE --peak-thresh=$PEAK_THRESH --max-features=$MAX_FEATURES --max-orientations=2 "$IMAGE" "$OUTPUT_DIR/sift_rust" > /tmp/sift_rust.log 2>&1
    END=$(date +%s.%N)
    TIME=$(calc_time "$START" "$END")
    FEATURES=$(grep "Extracted" /tmp/sift_rust.log | grep -oE '[0-9]+' | head -1)
    [ -z "$FEATURES" ] && FEATURES=$(grep "Features:" /tmp/sift_rust.log | grep -oE '[0-9]+' | tail -1)
    [ -z "$FEATURES" ] && FEATURES="N/A"
    echo "Rust (from scratch)|$FEATURES|$TIME" >> "$RESULTS_FILE"
else
    echo -e "${RED}Rust implementation not found${NC}"
    echo "Rust (from scratch)|N/A|N/A" >> "$RESULTS_FILE"
fi

echo ""
echo "=============================================="
echo "Results"
echo "=============================================="
echo ""

# Print table header
printf "+----------------------+------------+------------+\n"
printf "| %-20s | %-10s | %-10s |\n" "Implementation" "Features" "Time (s)"
printf "+----------------------+------------+------------+\n"

# Print results from file
while IFS='|' read -r name features time; do
    printf "| %-20s | %10s | %10s |\n" "$name" "$features" "$time"
done < "$RESULTS_FILE"

printf "+----------------------+------------+------------+\n"

echo ""
echo "Output files saved to: $OUTPUT_DIR/"
echo ""

# Print summary statistics
echo "=============================================="
echo "Summary"
echo "=============================================="
echo ""

# Find fastest and most features
FASTEST_NAME=""
FASTEST_TIME="999999"
MOST_FEATURES_NAME=""
MOST_FEATURES=0

while IFS='|' read -r name features time; do
    if [ "$time" != "N/A" ]; then
        IS_FASTER=$(awk "BEGIN {print ($time < $FASTEST_TIME) ? 1 : 0}")
        if [ "$IS_FASTER" -eq 1 ]; then
            FASTEST_TIME=$time
            FASTEST_NAME=$name
        fi
    fi
    if [ "$features" != "N/A" ] && [ "$features" -gt "$MOST_FEATURES" ]; then
        MOST_FEATURES=$features
        MOST_FEATURES_NAME=$name
    fi
done < "$RESULTS_FILE"

if [ -n "$FASTEST_NAME" ]; then
    echo -e "Fastest: ${GREEN}$FASTEST_NAME${NC} (${FASTEST_TIME}s)"
fi
if [ -n "$MOST_FEATURES_NAME" ]; then
    echo -e "Most features: ${GREEN}$MOST_FEATURES_NAME${NC} ($MOST_FEATURES)"
fi

echo ""
