#!/bin/bash
#
# SIFT Feature Extraction Script for COLMAP
# 
# This script:
# 1. Installs required build dependencies
# 2. Configures and builds the sift_extract tool
# 3. Extracts SIFT features from sample images
#
# Usage: ./extract_sift.sh [--skip-deps] [--skip-build]
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLMAP_DIR="${SCRIPT_DIR}/colmap"
BUILD_DIR="${COLMAP_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/output/sift"
MAX_FEATURES=1000

# Parse arguments
SKIP_DEPS=false
SKIP_BUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--skip-deps] [--skip-build]"
            echo ""
            echo "Options:"
            echo "  --skip-deps    Skip installing dependencies (use if already installed)"
            echo "  --skip-build   Skip building (use if already built)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "COLMAP SIFT Feature Extraction"
echo "=============================================="
echo ""

# Step 1: Install dependencies
if [ "$SKIP_DEPS" = false ]; then
    echo "[Step 1/4] Installing build dependencies..."
    echo ""
    
    apt-get update -qq
    
    # Build tools
    apt-get install -y -qq \
        cmake \
        build-essential \
        ninja-build \
        git
    
    # Core dependencies
    apt-get install -y -qq \
        libeigen3-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libsqlite3-dev \
        libboost-graph-dev \
        libboost-program-options-dev \
        libboost-system-dev
    
    # Image I/O
    apt-get install -y -qq \
        libopenimageio-dev \
        openimageio-tools
    
    # Linear algebra (Ceres dependencies)
    apt-get install -y -qq \
        libsuitesparse-dev \
        libmetis-dev \
        libceres-dev
    
    # OpenCV (required by OpenImageIO on some systems)
    apt-get install -y -qq \
        libopencv-dev \
        libglew-dev
    
    echo ""
    echo "Dependencies installed successfully."
else
    echo "[Step 1/4] Skipping dependency installation (--skip-deps)"
fi

echo ""

# Step 2: Configure CMake
if [ "$SKIP_BUILD" = false ]; then
    echo "[Step 2/4] Configuring CMake..."
    echo ""
    
    mkdir -p "${BUILD_DIR}"
    
    cd "${BUILD_DIR}"
    cmake .. \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGUI_ENABLED=OFF \
        -DCUDA_ENABLED=OFF \
        -DOPENGL_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        -DMVS_ENABLED=OFF \
        -DONNX_ENABLED=OFF \
        -DCCACHE_ENABLED=OFF \
        -DOPENMP_ENABLED=ON \
        -DSIMD_ENABLED=ON \
        -DCGAL_ENABLED=OFF
    
    echo ""
    echo "CMake configuration complete."
else
    echo "[Step 2/4] Skipping CMake configuration (--skip-build)"
fi

echo ""

# Step 3: Build sift_extract
if [ "$SKIP_BUILD" = false ]; then
    echo "[Step 3/4] Building sift_extract tool..."
    echo ""
    
    cd "${BUILD_DIR}"
    ninja sift_extract
    
    echo ""
    echo "Build complete."
else
    echo "[Step 3/4] Skipping build (--skip-build)"
fi

echo ""

# Step 4: Run feature extraction
echo "[Step 4/4] Extracting SIFT features from sample images..."
echo ""

SIFT_EXTRACT="${BUILD_DIR}/src/colmap/tools/sift_extract"

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
            grep -E "(Features extracted|Image:)" || true
    fi
done

echo ""

# Extract features from cars2 dataset (first 3 images)
echo "--- Processing cars2 dataset ---"
for img in "${SCRIPT_DIR}/data/cars2/cars2_0"{1..3}.jpg; do
    if [ -f "$img" ]; then
        echo "Extracting: $(basename "$img")"
        "${SIFT_EXTRACT}" --max_features ${MAX_FEATURES} "$img" "${OUTPUT_DIR}" 2>&1 | \
            grep -E "(Features extracted|Image:)" || true
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
echo "The sift_extract tool is available at:"
echo "  ${SIFT_EXTRACT}"
echo ""
echo "Usage:"
echo "  ${SIFT_EXTRACT} [options] <image_path> <output_dir>"
echo ""
echo "Options:"
echo "  --max_features N    Maximum number of features (default: 8192)"
echo "  --first_octave N    First octave (default: -1)"
echo "  --num_octaves N     Number of octaves (default: 4)"
echo "  --peak_thresh F     Peak threshold (default: 0.02/3)"
echo "  --edge_thresh F     Edge threshold (default: 10.0)"
echo "  --normalization N   Normalization: L1_ROOT or L2 (default: L1_ROOT)"
echo "  --upright           Fix orientation to 0 (upright features)"
