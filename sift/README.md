# Standalone SIFT Feature Extractor

A minimal, standalone SIFT feature extraction tool extracted from COLMAP.

## Features

- **CPU-only SIFT extraction** using VLFeat implementation
- **No external dependencies** beyond Eigen3 (header-only)
- **Simple image loading** using stb_image (single-header library)
- **Multiple output formats**: COLMAP text format, CSV, and optional binary

## Directory Structure

```
sift/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── build_sift.sh           # Build script
├── src/
│   ├── main.cc             # CLI tool
│   ├── sift_extractor.cc   # SIFT implementation
│   ├── sift_extractor.h    # SIFT interface
│   └── feature_types.h     # Feature types
└── thirdparty/
    ├── vlfeat/             # VLFeat SIFT core (BSD license)
    └── stb/
        └── stb_image.h     # Image loading (public domain)
```

## Dependencies

- **CMake** >= 3.12
- **C++17** compiler (GCC, Clang, or MSVC)
- **Eigen3** (header-only)

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SIMD_ENABLED` | ON | Enable SSE2 optimizations |
| `AVX_ENABLED` | OFF | Enable AVX optimizations (may reduce compatibility) |

```bash
cmake .. -DSIMD_ENABLED=OFF  # Disable SIMD for maximum compatibility
cmake .. -DAVX_ENABLED=ON    # Enable AVX for better performance
```

## Usage

```bash
./sift_extract [options] <image_path> <output_dir>
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max_features N` | 8192 | Maximum number of features |
| `--first_octave N` | -1 | First octave (-1 upsamples image) |
| `--num_octaves N` | 4 | Number of octaves |
| `--octave_levels N` | 3 | Levels per octave |
| `--peak_thresh F` | 0.00667 | Peak threshold for detection |
| `--edge_thresh F` | 10.0 | Edge threshold for detection |
| `--max_orientations N` | 2 | Max orientations per keypoint |
| `--normalization N` | L1_ROOT | Descriptor normalization (L1_ROOT or L2) |
| `--upright` | - | Fix orientation to 0 (upright features) |
| `--binary` | - | Also output binary descriptor file |

### Output Files

| File | Format |
|------|--------|
| `<name>_keypoints.txt` | COLMAP text format |
| `<name>_keypoints.csv` | CSV format (x, y, scale, orientation) |
| `<name>_descriptors.bin` | Binary descriptors (with --binary) |

### Output Format

**Text format (COLMAP compatible):**
```
<num_features> 128
<x> <y> <scale> <orientation> <d1> <d2> ... <d128>
...
```

**CSV format:**
```csv
x,y,scale,orientation
<x>,<y>,<scale>,<orientation>
...
```

**Binary format:**
```
int32: num_features
int32: descriptor_dim (128)
uint8[num_features * 128]: descriptors
```

## Example

```bash
# Extract up to 1000 features from an image
./sift_extract --max_features 1000 image.jpg output/

# Extract upright features with L2 normalization
./sift_extract --upright --normalization L2 image.jpg output/

# Extract features with binary output
./sift_extract --binary --max_features 5000 image.jpg output/
```

## Implementation Details

### SIFT Algorithm

This implementation uses VLFeat's SIFT algorithm, which follows David Lowe's original paper:

1. **Scale-space construction**: Gaussian pyramid with configurable octaves and levels
2. **Keypoint detection**: Difference of Gaussians (DoG) extrema detection
3. **Orientation assignment**: Histogram of gradient orientations
4. **Descriptor computation**: 128-dimensional gradient histogram

### VLFeat Files

The following VLFeat source files are included:

| File | Purpose |
|------|---------|
| `sift.c/h` | Core SIFT algorithm |
| `generic.c/h` | Platform abstraction |
| `mathop.c/h` | Math operations |
| `random.c/h` | Random number generation |
| `host.c/h` | Host utilities |
| `imopv.c/h` | Image operations |
| `scalespace.c/h` | Scale space construction |
| `mathop_sse2.c/h` | SSE2 optimizations |
| `imopv_sse2.c/h` | SSE2 image operations |

### Differences from COLMAP

1. **Simplified dependencies**: No glog, OpenImageIO, or other COLMAP dependencies
2. **stb_image**: Uses stb_image instead of OpenImageIO for image loading
3. **Standalone**: Self-contained, no need to build the full COLMAP project

## License

- **COLMAP wrapper code**: BSD 3-Clause (same as COLMAP)
- **VLFeat**: BSD license (see thirdparty/vlfeat/)
- **stb_image**: Public domain (see thirdparty/stb/stb_image.h)

## References

- Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
- VLFeat: http://www.vlfeat.org/
- COLMAP: https://colmap.github.io/
