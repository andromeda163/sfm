# sift-wgpu Wrapper

A minimal wrapper around `sift-wgpu` that adds VLFeat/COLMAP-compatible parameters.

## Features Added

This wrapper extends `sift-wgpu` with the following features:

1. **`first_octave` parameter**: When set to -1, the image is upsampled 2x before processing, matching VLFeat's behavior for detecting more features at fine scales.

2. **`max_features` limiting**: Keeps only the largest scale features when the limit is exceeded.

3. **`max_orientations` limiting**: Limits the number of orientations per keypoint location.

4. **Proper descriptor format**: Converts descriptors to UBC format for compatibility with other tools.

## Usage

```bash
./sift_extract_wgpu [OPTIONS] <IMAGE_PATH> <OUTPUT_DIR>

Options:
  -m, --max-features <MAX_FEATURES>  Maximum number of features [default: 8192]
      --first-octave <FIRST_OCTAVE>  First octave (-1 = upsample) [default: -1]
      --num-octaves <NUM_OCTAVES>    Number of octaves [default: 4]
      --octave-levels <OCTAVE_LEVELS> Octave levels/intervals [default: 3]
      --peak-thresh <PEAK_THRESH>    Peak threshold [default: 0.02/levels]
      --edge-thresh <EDGE_THRESH>    Edge threshold [default: 10.0]
      --max-orientations <MAX_ORIENTATIONS> Max orientations [default: 2]
      --upright                      Fix orientation to 0
```

## Example

```bash
# Extract features with upsampling (first_octave=-1)
./sift_extract_wgpu --max-features 1000 --first-octave=-1 image.jpg output/

# Extract features without upsampling (first_octave=0)
./sift_extract_wgpu --max-features 1000 --first-octave=0 image.jpg output/
```

## Comparison with Other Implementations

| Implementation | first_octave=0 | first_octave=-1 |
|---------------|----------------|-----------------|
| C++ (VLFeat)  | ~1357 features | ~5188 features  |
| sift-wgpu     | ~519 features  | N/A             |
| This wrapper  | ~2002 features | ~8418 features  |
| Rust from scratch | ~503 features | ~9337 features |

Note: The feature counts differ due to variations in:
- Gaussian blur implementation
- Peak/edge threshold handling
- Keypoint refinement algorithm

## Building

```bash
cd sift-wgpu
cargo build --release
```

## Output Format

The wrapper outputs:
- `<name>_keypoints.txt`: COLMAP-compatible text format
- `<name>_keypoints.csv`: CSV format with x, y, scale, orientation
- `<name>_descriptors.bin`: Binary descriptor file
