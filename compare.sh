#!/bin/bash
# Compare SIFT features extracted by COLMAP, Standalone C++, and sift-wgpu (Rust)
# Includes timing benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLMAP_OUTPUT="${SCRIPT_DIR}/output/sift"
CUSTOM_OUTPUT="${SCRIPT_DIR}/output/sift_custom"
WGPU_OUTPUT="${SCRIPT_DIR}/output/sift_wgpu"
DATA_DIR="${SCRIPT_DIR}/data"
VIS_OUTPUT="${SCRIPT_DIR}/output/comparison"
BENCHMARK_OUTPUT="${SCRIPT_DIR}/output/benchmark"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Benchmark parameters
MAX_FEATURES=${MAX_FEATURES:-1000}
WARMUP=${WARMUP:-2}
ITERATIONS=${ITERATIONS:-5}
IMAGE_LIMIT=${IMAGE_LIMIT:-10}

echo "=============================================="
echo "SIFT Feature Comparison & Benchmark Tool"
echo "=============================================="
echo ""
echo "COLMAP output:      ${COLMAP_OUTPUT}"
echo "Custom output:      ${CUSTOM_OUTPUT}"
echo "sift-wgpu output:   ${WGPU_OUTPUT}"
echo "Visual output:      ${VIS_OUTPUT}"
echo "Benchmark output:   ${BENCHMARK_OUTPUT}"
echo ""
echo "Parameters:"
echo "  Max features:     ${MAX_FEATURES}"
echo "  Warmup:           ${WARMUP}"
echo "  Iterations:       ${ITERATIONS}"
echo "  Image limit:      ${IMAGE_LIMIT}"
echo ""

# Create output directories
mkdir -p "${VIS_OUTPUT}"
mkdir -p "${BENCHMARK_OUTPUT}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required"
    exit 1
fi

# Function to time a command
time_command() {
    local start end duration
    start=$(date +%s.%N)
    "$@" > /dev/null 2>&1
    end=$(date +%s.%N)
    duration=$(awk "BEGIN {printf \"%.2f\", ($end - $start) * 1000}")
    echo "$duration"
}

# Function to run benchmark for an implementation
run_benchmark() {
    local name=$1
    local binary=$2
    local output_dir=$3
    shift 3
    
    echo -e "${BLUE}Benchmarking ${name}...${NC}"
    
    if [ ! -f "${binary}" ]; then
        echo -e "  ${YELLOW}Binary not found: ${binary}${NC}"
        return 1
    fi
    
    mkdir -p "${output_dir}"
    
    local count=0
    local total_time=0
    local total_features=0
    
    # Process images from bear01 dataset
    for img in "${DATA_DIR}/bear01/bear01_000"{1..9}.jpg "${DATA_DIR}/bear01/bear01_001"{0..${IMAGE_LIMIT}}.jpg; do
        if [ -f "$img" ] && [ $count -lt $IMAGE_LIMIT ]; then
            local img_name=$(basename "$img" .jpg)
            
            # Warmup runs
            for i in $(seq 1 $WARMUP); do
                "${binary}" --max_features ${MAX_FEATURES} "$img" "${output_dir}" > /dev/null 2>&1 || true
            done
            
            # Timed runs
            local run_times=""
            for i in $(seq 1 $ITERATIONS); do
                local ms=$(time_command "${binary}" --max_features ${MAX_FEATURES} "$img" "${output_dir}")
                run_times="$run_times $ms"
            done
            
            # Calculate average
            local avg_time=$(echo "$run_times" | tr ' ' '\n' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
            
            # Get feature count
            local txt_file="${output_dir}/${img_name}_keypoints.txt"
            local features=0
            if [ -f "$txt_file" ]; then
                features=$(head -1 "$txt_file" | cut -d' ' -f1)
            fi
            
            printf "  %-20s %8.2f ms  %6d features\n" "$img_name" "$avg_time" "$features"
            
            total_time=$(awk "BEGIN {printf \"%.2f\", $total_time + $avg_time}")
            total_features=$((total_features + features))
            count=$((count + 1))
        fi
    done
    
    if [ $count -gt 0 ]; then
        local avg_total=$(awk "BEGIN {printf \"%.2f\", $total_time / $count}")
        echo -e "  ${GREEN}Average: ${avg_total} ms, ${total_features} total features${NC}"
    fi
    
    echo ""
    return 0
}

# Run benchmarks
echo "=============================================="
echo "RUNNING BENCHMARKS"
echo "=============================================="
echo ""

# Benchmark COLMAP
COLMAP_BIN="${SCRIPT_DIR}/colmap/build/src/colmap/tools/sift_extract"
if [ -f "${COLMAP_BIN}" ]; then
    run_benchmark "COLMAP (C++/VLFeat)" "${COLMAP_BIN}" "${COLMAP_OUTPUT}"
else
    echo -e "${YELLOW}COLMAP not built. Run extract_sift.sh first.${NC}"
    echo ""
fi

# Benchmark Standalone C++
STANDALONE_BIN="${SCRIPT_DIR}/sift/build/sift_extract"
if [ -f "${STANDALONE_BIN}" ]; then
    run_benchmark "Standalone C++ (VLFeat)" "${STANDALONE_BIN}" "${CUSTOM_OUTPUT}"
else
    echo -e "${YELLOW}Standalone SIFT not built. Run extract_sift_custom.sh first.${NC}"
    echo ""
fi

# Benchmark sift-wgpu (Rust)
WGPU_BIN="${SCRIPT_DIR}/benchmark/target/release/sift-benchmark"
if [ -f "${WGPU_BIN}" ]; then
    echo -e "${BLUE}Benchmarking sift-wgpu (Rust)...${NC}"
    "${WGPU_BIN}" \
        --input "${DATA_DIR}/bear01" \
        --output "${BENCHMARK_OUTPUT}" \
        --max-features ${MAX_FEATURES} \
        --warmup ${WARMUP} \
        --iterations ${ITERATIONS} \
        --limit ${IMAGE_LIMIT} \
        --format text 2>&1 || echo -e "${YELLOW}sift-wgpu benchmark failed${NC}"
    echo ""
else
    echo -e "${YELLOW}sift-wgpu not built. Run 'cd benchmark && cargo build --release'.${NC}"
    echo ""
fi

# Run feature comparison
echo "=============================================="
echo "FEATURE COMPARISON"
echo "=============================================="
echo ""

# Install required packages
pip3 install numpy pillow matplotlib --quiet 2>/dev/null || pip install numpy pillow matplotlib --quiet 2>/dev/null || true

# Create Python comparison script
cat > /tmp/compare_sift.py << 'PYTHON_EOF'
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import time

def parse_keypoints_txt(filepath):
    """Parse COLMAP format keypoints file"""
    keypoints = []
    descriptors = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # First line: num_features descriptor_dim
    header = lines[0].strip().split()
    num_features = int(header[0])
    descriptor_dim = int(header[1])
    
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 4 + descriptor_dim:
            x = float(parts[0])
            y = float(parts[1])
            scale = float(parts[2])
            orientation = float(parts[3])
            desc = np.array([int(p) for p in parts[4:4+descriptor_dim]], dtype=np.uint8)
            keypoints.append((x, y, scale, orientation))
            descriptors.append(desc)
    
    return keypoints, np.array(descriptors)

def compute_l2_distance(desc1, desc2):
    """Compute L2 distance between two descriptors"""
    return np.linalg.norm(desc1.astype(np.float32) - desc2.astype(np.float32))

def match_features(kp1, desc1, kp2, desc2, threshold=0.8):
    """Simple feature matching using L2 distance ratio test"""
    matches = []
    
    for i, (k1, d1) in enumerate(zip(kp1, desc1)):
        distances = []
        for j, (k2, d2) in enumerate(zip(kp2, desc2)):
            dist = compute_l2_distance(d1, d2)
            distances.append((dist, j))
        
        distances.sort()
        if len(distances) >= 2:
            best_dist, best_idx = distances[0]
            second_dist = distances[1][0]
            
            # Ratio test
            if best_dist < threshold * second_dist:
                matches.append((i, best_idx, best_dist))
    
    return matches

def draw_keypoints(image_path, keypoints, output_path, title, color=(0, 255, 0)):
    """Draw keypoints on image and save"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for i, (x, y, scale, orientation) in enumerate(keypoints):
        # Draw keypoint as circle
        radius = max(2, int(scale * 3))
        
        # Draw circle
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                     outline=color, width=2)
        
        # Draw orientation line
        dx = radius * 1.5 * math.cos(orientation)
        dy = radius * 1.5 * math.sin(orientation)
        draw.line([(x, y), (x + dx, y + dy)], fill=color, width=2)
    
    # Add title and count
    draw.text((10, 10), f"{title}: {len(keypoints)} features", fill=(255, 255, 255), font=font)
    
    img.save(output_path)
    return img

def create_comparison_image(image_path, kp1, kp2, matches, output_path, name):
    """Create side-by-side comparison image with matches"""
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    # Create combined image (2x width)
    combined = Image.new('RGB', (w * 2 + 50, h + 100), (40, 40, 40))
    combined.paste(img, (0, 50))
    combined.paste(img, (w + 50, 50))
    
    draw = ImageDraw.Draw(combined)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_title = font
    
    # Draw titles
    draw.text((10, 10), f"COLMAP: {len(kp1)} features", fill=(0, 255, 0), font=font_title)
    draw.text((w + 60, 10), f"Standalone: {len(kp2)} features", fill=(255, 165, 0), font=font_title)
    
    # Draw keypoints on left (COLMAP - green)
    for x, y, scale, orientation in kp1:
        radius = max(2, int(scale * 3))
        draw.ellipse([x - radius, y - radius + 50, x + radius, y + radius + 50], 
                     outline=(0, 255, 0), width=1)
    
    # Draw keypoints on right (Standalone - orange)
    for x, y, scale, orientation in kp2:
        radius = max(2, int(scale * 3))
        draw.ellipse([x + w + 50 - radius, y - radius + 50, x + w + 50 + radius, y + radius + 50], 
                     outline=(255, 165, 0), width=1)
    
    # Draw match lines (every 5th match for clarity)
    for idx, (i, j, dist) in enumerate(matches[::5]):
        x1, y1 = kp1[i][0], kp1[i][1] + 50
        x2, y2 = kp2[j][0] + w + 50, kp2[j][1] + 50
        
        # Color based on distance
        intensity = max(0, min(255, int(255 * (1 - dist / 512))))
        color = (intensity, intensity, 255)
        
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
    
    # Add stats
    stats_text = f"Matches: {len(matches)} | Match rate: {100*len(matches)/max(len(kp1), len(kp2)):.1f}%"
    draw.text((10, h + 60), stats_text, fill=(255, 255, 255), font=font)
    
    combined.save(output_path)
    return combined

def compare_outputs(colmap_dir, custom_dir, vis_dir, data_dir):
    """Compare all matching output files"""
    results = []
    
    # Find matching files
    colmap_files = sorted([f for f in os.listdir(colmap_dir) if f.endswith('_keypoints.txt')])
    
    for colmap_file in colmap_files:
        base_name = colmap_file.replace('_keypoints.txt', '')
        custom_file = base_name + '_keypoints.txt'
        
        colmap_path = os.path.join(colmap_dir, colmap_file)
        custom_path = os.path.join(custom_dir, custom_file)
        
        if not os.path.exists(custom_path):
            print(f"  Skipping {base_name}: no matching custom output")
            continue
        
        print(f"\n{'='*50}")
        print(f"Comparing: {base_name}")
        print(f"{'='*50}")
        
        # Load keypoints and descriptors
        kp_colmap, desc_colmap = parse_keypoints_txt(colmap_path)
        kp_custom, desc_custom = parse_keypoints_txt(custom_path)
        
        print(f"  COLMAP features:    {len(kp_colmap)}")
        print(f"  Custom features:    {len(kp_custom)}")
        print(f"  Feature count diff: {abs(len(kp_colmap) - len(kp_custom))}")
        
        # Match features
        matches = match_features(kp_colmap, desc_colmap, kp_custom, desc_custom)
        
        if len(matches) > 0:
            match_distances = [m[2] for m in matches]
            avg_dist = np.mean(match_distances)
            min_dist = np.min(match_distances)
            max_dist = np.max(match_distances)
            
            print(f"\n  Feature Matching:")
            print(f"    Matches:          {len(matches)}")
            print(f"    Match rate:       {100*len(matches)/max(len(kp_colmap), len(kp_custom)):.1f}%")
            print(f"    Avg L2 distance:  {avg_dist:.2f}")
            print(f"    Min L2 distance:  {min_dist:.2f}")
            print(f"    Max L2 distance:  {max_dist:.2f}")
        
        # Find corresponding image
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(data_dir, 'bear01', base_name + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break
            test_path = os.path.join(data_dir, 'cars2', base_name.replace('cars2_', 'cars2_') + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break
            test_path = os.path.join(data_dir, 'horses03', base_name.replace('horses03_', 'horses03_') + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        # Create visualizations
        if image_path and os.path.exists(image_path):
            # Draw COLMAP keypoints
            colmap_vis = os.path.join(vis_dir, base_name + '_colmap.png')
            draw_keypoints(image_path, kp_colmap, colmap_vis, "COLMAP", color=(0, 255, 0))
            
            # Draw Custom keypoints
            custom_vis = os.path.join(vis_dir, base_name + '_custom.png')
            draw_keypoints(image_path, kp_custom, custom_vis, "Standalone", color=(255, 165, 0))
            
            # Create comparison image
            comparison_vis = os.path.join(vis_dir, base_name + '_comparison.png')
            create_comparison_image(image_path, kp_colmap, kp_custom, matches, comparison_vis, base_name)
            
            print(f"\n  Visualizations saved:")
            print(f"    {colmap_vis}")
            print(f"    {custom_vis}")
            print(f"    {comparison_vis}")
        
        results.append({
            'name': base_name,
            'colmap_count': len(kp_colmap),
            'custom_count': len(kp_custom),
            'matches': len(matches),
            'avg_distance': avg_dist if matches else float('nan')
        })
    
    return results

def main():
    if len(sys.argv) < 4:
        print("Usage: compare_sift.py <colmap_dir> <custom_dir> <vis_dir> [data_dir]")
        sys.exit(1)
    
    colmap_dir = sys.argv[1]
    custom_dir = sys.argv[2]
    vis_dir = sys.argv[3]
    data_dir = sys.argv[4] if len(sys.argv) > 4 else ""
    
    os.makedirs(vis_dir, exist_ok=True)
    
    results = compare_outputs(colmap_dir, custom_dir, vis_dir, data_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Image':<20} {'COLMAP':>8} {'Custom':>8} {'Diff':>6} {'Matches':>8} {'AvgDist':>8}")
    print("-" * 60)
    
    for r in results:
        diff = abs(r['colmap_count'] - r['custom_count'])
        avg_dist = f"{r['avg_distance']:.1f}" if not math.isnan(r['avg_distance']) else "N/A"
        print(f"{r['name']:<20} {r['colmap_count']:>8} {r['custom_count']:>8} {diff:>6} {r['matches']:>8} {avg_dist:>8}")
    
    # Overall stats
    total_colmap = sum(r['colmap_count'] for r in results)
    total_custom = sum(r['custom_count'] for r in results)
    total_matches = sum(r['matches'] for r in results)
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_colmap:>8} {total_custom:>8} {abs(total_colmap - total_custom):>6} {total_matches:>8}")

if __name__ == "__main__":
    main()
PYTHON_EOF

# Run comparison if both COLMAP and Standalone outputs exist
if [ -d "${COLMAP_OUTPUT}" ] && [ -d "${CUSTOM_OUTPUT}" ]; then
    echo "Running feature comparison..."
    python3 /tmp/compare_sift.py "${COLMAP_OUTPUT}" "${CUSTOM_OUTPUT}" "${VIS_OUTPUT}" "${DATA_DIR}"
else
    echo -e "${YELLOW}Skipping feature comparison - missing output directories${NC}"
fi

echo ""
echo "=============================================="
echo "Benchmark and Comparison Complete!"
echo "=============================================="
echo ""
echo "Visualizations saved to: ${VIS_OUTPUT}"
echo "Benchmark results saved to: ${BENCHMARK_OUTPUT}"
echo ""
echo "To view the comparison images:"
echo "  ls ${VIS_OUTPUT}/*.png"
echo ""
echo "To run the Python benchmark script directly:"
echo "  python3 ${SCRIPT_DIR}/benchmark_sift.py --input ${DATA_DIR}/bear01 --output ${BENCHMARK_OUTPUT}"
