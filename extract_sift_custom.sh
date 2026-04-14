#!/bin/bash
#
# SIFT Feature Extraction Script for Standalone SIFT Tool
# 
# This script extracts SIFT features using the standalone sift_extract tool
# and produces output comparable to extract_sift.sh
#
# Usage: ./extract_sift_custom.sh [--skip-build]
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIFT_DIR="${SCRIPT_DIR}/sift"
BUILD_DIR="${SIFT_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/output/sift_custom"
MAX_FEATURES=1000

# Parse arguments
SKIP_BUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--skip-build]"
            echo ""
            echo "Options:"
            echo "  --skip-build    Skip building (use if already built)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Standalone SIFT Feature Extraction"
echo "=============================================="
echo ""

# Step 1: Build (if needed)
if [ "$SKIP_BUILD" = false ]; then
    echo "[Step 1/2] Building standalone SIFT tool..."
    echo ""
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DSIMD_ENABLED=ON \
        -DAVX_ENABLED=OFF
    
    make -j$(nproc)
    
    echo ""
    echo "Build complete."
else
    echo "[Step 1/2] Skipping build (--skip-build)"
fi

echo ""

# Step 2: Run feature extraction
echo "[Step 2/2] Extracting SIFT features from sample images..."
echo ""

SIFT_EXTRACT="${BUILD_DIR}/sift_extract"

if [ ! -f "${SIFT_EXTRACT}" ]; then
    echo "ERROR: sift_extract binary not found at ${SIFT_EXTRACT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Extract features from bear01 dataset (first 5 images)
echo "--- Processing bear01 dataset ---"
for img in "${SCRIPT_DIR}/data/bear01/bear01_000"{1..5}.jpg; do
    if [ -f "$img" ]; then
        echo "Extracting: $(basename "$img")"
        "${SIFT_EXTRACT}" --max_features ${MAX_FEATURES} "$img" "${OUTPUT_DIR}" 2>&1 | \
            grep -E "(Features:|Image:)" || true
    fi
done

echo ""

# Extract features from cars2 dataset (first 3 images)
echo "--- Processing cars2 dataset ---"
for img in "${SCRIPT_DIR}/data/cars2/cars2_0"{1..3}.jpg; do
    if [ -f "$img" ]; then
        echo "Extracting: $(basename "$img")"
        "${SIFT_EXTRACT}" --max_features ${MAX_FEATURES} "$img" "${OUTPUT_DIR}" 2>&1 | \
            grep -E "(Features:|Image:)" || true
    fi
done

echo ""

# Show results
echo "=============================================="
echo "Results"
echo "=============================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -lh "${OUTPUT_DIR}/" 2>/dev/null || echo "  (no files)"
echo ""

# Show sample output
echo "Sample output (first 5 keypoints from bear01_0001):"
echo "---"
head -6 "${OUTPUT_DIR}/bear01_0001_keypoints.txt" 2>/dev/null || echo "  (file not found)"
echo ""

echo "CSV format (first 5 keypoints):"
head -6 "${OUTPUT_DIR}/bear01_0001_keypoints.csv" 2>/dev/null || echo "  (file not found)"
echo ""

echo "=============================================="
echo "Done!"
echo "=============================================="
echo ""
echo "The standalone sift_extract tool is available at:"
echo "  ${SIFT_EXTRACT}"
echo ""
echo "Usage:"
echo "  ${SIFT_EXTRACT} [options] <image_path> <output_dir>"
echo ""
echo "Options:"
echo "  --max_features N    Maximum number of features (default: 8192)"
echo "  --first_octave N    First octave (default: -1)"
echo "  --num_octaves N     Number of octaves (default: 4)"
echo "  --octave_levels N   Levels per octave (default: 3)"
echo "  --peak_thresh F     Peak threshold (default: 0.00667)"
echo "  --edge_thresh F     Edge threshold (default: 10.0)"
echo "  --max_orientations N Max orientations per keypoint (default: 2)"
echo "  --normalization N   Normalization: L1_ROOT or L2 (default: L1_ROOT)"
echo "  --upright           Fix orientation to 0 (upright features)"
echo "  --binary            Also output binary descriptor file"
