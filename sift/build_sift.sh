#!/bin/bash
#
# Build script for Standalone SIFT Feature Extractor
#
# Usage: ./build_sift.sh [--clean]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Parse arguments
CLEAN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Building Standalone SIFT Feature Extractor"
echo "=============================================="
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
echo "Configuring..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSIMD_ENABLED=ON \
    -DAVX_ENABLED=OFF

echo ""

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Binary: ${BUILD_DIR}/sift_extract"
echo ""
echo "Usage:"
echo "  ${BUILD_DIR}/sift_extract --help"
echo "  ${BUILD_DIR}/sift_extract <image_path> <output_dir>"
