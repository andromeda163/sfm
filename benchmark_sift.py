#!/usr/bin/env python3
"""
SIFT Feature Extraction Benchmark and Comparison Tool

Compares SIFT implementations:
1. COLMAP (C++ with VLFeat)
2. Standalone SIFT (C++ with VLFeat)
3. sift-wgpu (Rust)

Outputs timing benchmarks and feature matching comparisons.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional
import math

# Try to import optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class Keypoint:
    x: float
    y: float
    scale: float
    orientation: float
    descriptor: Optional[List[int]] = None


@dataclass
class ExtractionResult:
    image_name: str
    width: int
    height: int
    num_features: int
    extraction_time_ms: float
    keypoints: List[Keypoint]
    implementation: str


@dataclass
class BenchmarkResult:
    image_name: str
    implementation: str
    width: int
    height: int
    num_features: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_mpixels_per_sec: float


def parse_keypoints_txt(filepath: str) -> Tuple[List[Keypoint], int, int]:
    """Parse COLMAP format keypoints file"""
    keypoints = []
    width, height = 0, 0
    
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
            desc = [int(p) for p in parts[4:4+descriptor_dim]]
            keypoints.append(Keypoint(x, y, scale, orientation, desc))
    
    return keypoints, num_features, descriptor_dim


def parse_csv_keypoints(filepath: str) -> List[Keypoint]:
    """Parse CSV format keypoints file"""
    keypoints = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            keypoints.append(Keypoint(
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3])
            ))
    
    return keypoints


def benchmark_colmap_sift(image_path: str, output_dir: str, max_features: int = 1000,
                          warmup: int = 2, iterations: int = 5) -> Optional[BenchmarkResult]:
    """Benchmark COLMAP's SIFT extraction"""
    sift_extract = Path(__file__).parent / "colmap" / "build" / "src" / "colmap" / "tools" / "sift_extract"
    
    if not sift_extract.exists():
        print(f"  COLMAP sift_extract not found at {sift_extract}")
        return None
    
    image_name = Path(image_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    times = []
    num_features = 0
    
    # Warmup
    for _ in range(warmup):
        result = subprocess.run(
            [str(sift_extract), "--max_features", str(max_features), image_path, str(output_path)],
            capture_output=True, text=True
        )
    
    # Timed runs
    for _ in range(iterations):
        start = time.perf_counter()
        result = subprocess.run(
            [str(sift_extract), "--max_features", str(max_features), image_path, str(output_path)],
            capture_output=True, text=True
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    # Get feature count
    txt_file = output_path / f"{image_name}_keypoints.txt"
    if txt_file.exists():
        keypoints, num_features, _ = parse_keypoints_txt(str(txt_file))
    
    # Get image dimensions
    if HAS_PIL:
        with Image.open(image_path) as img:
            width, height = img.size
    else:
        width, height = 0, 0
    
    return BenchmarkResult(
        image_name=image_name,
        implementation="COLMAP",
        width=width,
        height=height,
        num_features=num_features,
        avg_time_ms=sum(times) / len(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        throughput_mpixels_per_sec=(width * height / 1_000_000) / (sum(times) / len(times) / 1000) if width > 0 else 0
    )


def benchmark_standalone_sift(image_path: str, output_dir: str, max_features: int = 1000,
                              warmup: int = 2, iterations: int = 5) -> Optional[BenchmarkResult]:
    """Benchmark standalone SIFT extraction"""
    sift_extract = Path(__file__).parent / "sift" / "build" / "sift_extract"
    
    if not sift_extract.exists():
        print(f"  Standalone sift_extract not found at {sift_extract}")
        return None
    
    image_name = Path(image_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    times = []
    num_features = 0
    
    # Warmup
    for _ in range(warmup):
        result = subprocess.run(
            [str(sift_extract), "--max_features", str(max_features), image_path, str(output_path)],
            capture_output=True, text=True
        )
    
    # Timed runs
    for _ in range(iterations):
        start = time.perf_counter()
        result = subprocess.run(
            [str(sift_extract), "--max_features", str(max_features), image_path, str(output_path)],
            capture_output=True, text=True
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    # Get feature count
    txt_file = output_path / f"{image_name}_keypoints.txt"
    if txt_file.exists():
        keypoints, num_features, _ = parse_keypoints_txt(str(txt_file))
    
    # Get image dimensions
    if HAS_PIL:
        with Image.open(image_path) as img:
            width, height = img.size
    else:
        width, height = 0, 0
    
    return BenchmarkResult(
        image_name=image_name,
        implementation="Standalone-C++",
        width=width,
        height=height,
        num_features=num_features,
        avg_time_ms=sum(times) / len(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        throughput_mpixels_per_sec=(width * height / 1_000_000) / (sum(times) / len(times) / 1000) if width > 0 else 0
    )


def benchmark_sift_wgpu(image_dir: str, output_dir: str, max_features: int = 1000,
                        warmup: int = 2, iterations: int = 5) -> List[BenchmarkResult]:
    """Benchmark sift-wgpu Rust implementation"""
    benchmark_bin = Path(__file__).parent / "benchmark" / "target" / "release" / "sift-benchmark"
    
    if not benchmark_bin.exists():
        print(f"  sift-wgpu benchmark not found at {benchmark_bin}")
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run the Rust benchmark
    result = subprocess.run(
        [
            str(benchmark_bin),
            "--input", image_dir,
            "--output", str(output_path),
            "--max-features", str(max_features),
            "--warmup", str(warmup),
            "--iterations", str(iterations),
            "--format", "json"
        ],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"  sift-wgpu benchmark failed: {result.stderr}")
        return []
    
    # Parse JSON output
    results = []
    json_file = output_path / "benchmark_results.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
            for item in data:
                results.append(BenchmarkResult(
                    image_name=item['image'],
                    implementation=item['implementation'],
                    width=item['width'],
                    height=item['height'],
                    num_features=item['num_features'],
                    avg_time_ms=item['avg_time_ms'],
                    min_time_ms=item['min_time_ms'],
                    max_time_ms=item['max_time_ms'],
                    throughput_mpixels_per_sec=item['throughput_mpixels_per_sec']
                ))
    
    return results


def compare_features(kp1: List[Keypoint], kp2: List[Keypoint], 
                     threshold: float = 0.8) -> List[Tuple[int, int, float]]:
    """Match features using L2 distance ratio test"""
    if not HAS_NUMPY or not kp1 or not kp2:
        return []
    
    matches = []
    desc1 = np.array([kp.descriptor for kp in kp1 if kp.descriptor], dtype=np.float32)
    desc2 = np.array([kp.descriptor for kp in kp2 if kp.descriptor], dtype=np.float32)
    
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    for i, d1 in enumerate(desc1):
        # Compute L2 distances to all descriptors in desc2
        distances = np.linalg.norm(desc2 - d1, axis=1)
        
        # Get two best matches
        if len(distances) >= 2:
            sorted_indices = np.argsort(distances)
            best_dist = distances[sorted_indices[0]]
            second_dist = distances[sorted_indices[1]]
            
            # Ratio test
            if best_dist < threshold * second_dist:
                matches.append((i, sorted_indices[0], best_dist))
    
    return matches


def draw_keypoints(image_path: str, keypoints: List[Keypoint], 
                   output_path: str, title: str, color: Tuple[int, int, int] = (0, 255, 0)):
    """Draw keypoints on image and save"""
    if not HAS_PIL:
        return
    
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for kp in keypoints:
        radius = max(2, int(kp.scale * 3))
        draw.ellipse([kp.x - radius, kp.y - radius, kp.x + radius, kp.y + radius],
                     outline=color, width=2)
        
        # Draw orientation line
        dx = radius * 1.5 * math.cos(kp.orientation)
        dy = radius * 1.5 * math.sin(kp.orientation)
        draw.line([(kp.x, kp.y), (kp.x + dx, kp.y + dy)], fill=color, width=2)
    
    draw.text((10, 10), f"{title}: {len(keypoints)} features", fill=(255, 255, 255), font=font)
    img.save(output_path)


def run_benchmark(args):
    """Run the full benchmark suite"""
    print("=" * 60)
    print("SIFT Feature Extraction Benchmark")
    print("=" * 60)
    print()
    
    all_results: List[BenchmarkResult] = []
    
    # Collect images
    image_dir = Path(args.input)
    images = []
    if image_dir.is_file():
        images = [str(image_dir)]
    elif image_dir.is_dir():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend([str(p) for p in image_dir.glob(ext)])
        images = sorted(images)
    
    if not images:
        print(f"No images found in {args.input}")
        return
    
    print(f"Images to process: {len(images)}")
    print(f"Max features: {args.max_features}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Timed iterations: {args.iterations}")
    print()
    
    # Create output directories
    output_dir = Path(args.output)
    colmap_output = output_dir / "colmap"
    standalone_output = output_dir / "standalone"
    wgpu_output = output_dir / "sift-wgpu"
    
    # Benchmark COLMAP
    if not args.skip_colmap:
        print("Benchmarking COLMAP...")
        for img in images[:args.limit]:
            result = benchmark_colmap_sift(
                img, str(colmap_output), args.max_features, args.warmup, args.iterations
            )
            if result:
                all_results.append(result)
                print(f"  {result.image_name}: {result.avg_time_ms:.2f}ms, {result.num_features} features")
        print()
    
    # Benchmark Standalone C++
    if not args.skip_standalone:
        print("Benchmarking Standalone C++...")
        for img in images[:args.limit]:
            result = benchmark_standalone_sift(
                img, str(standalone_output), args.max_features, args.warmup, args.iterations
            )
            if result:
                all_results.append(result)
                print(f"  {result.image_name}: {result.avg_time_ms:.2f}ms, {result.num_features} features")
        print()
    
    # Benchmark sift-wgpu
    if not args.skip_wgpu:
        print("Benchmarking sift-wgpu (Rust)...")
        wgpu_results = benchmark_sift_wgpu(
            args.input, str(wgpu_output), args.max_features, args.warmup, args.iterations
        )
        all_results.extend(wgpu_results)
        for r in wgpu_results:
            print(f"  {r.image_name}: {r.avg_time_ms:.2f}ms, {r.num_features} features")
        print()
    
    # Print summary
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Image':<25} {'Implementation':<15} {'Size':>10} {'Features':>10} {'Time (ms)':>12} {'MP/s':>10}")
    print("-" * 80)
    
    for r in sorted(all_results, key=lambda x: (x.image_name, x.implementation)):
        size = f"{r.width}x{r.height}" if r.width > 0 else "N/A"
        print(f"{r.image_name:<25} {r.implementation:<15} {size:>10} {r.num_features:>10} {r.avg_time_ms:>12.2f} {r.throughput_mpixels_per_sec:>10.2f}")
    
    # Print per-implementation summary
    print("-" * 80)
    impls = {}
    for r in all_results:
        if r.implementation not in impls:
            impls[r.implementation] = []
        impls[r.implementation].append(r)
    
    print(f"{'Implementation':<15} {'Images':>8} {'Avg Time':>12} {'Avg Features':>14} {'Avg MP/s':>12}")
    print("-" * 80)
    for impl, results in sorted(impls.items()):
        avg_time = sum(r.avg_time_ms for r in results) / len(results)
        avg_features = sum(r.num_features for r in results) / len(results)
        avg_throughput = sum(r.throughput_mpixels_per_sec for r in results) / len(results)
        print(f"{impl:<15} {len(results):>8} {avg_time:>12.2f} {avg_features:>14.1f} {avg_throughput:>12.2f}")
    
    # Save results to JSON
    json_output = output_dir / "benchmark_results.json"
    with open(json_output, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to: {json_output}")


def main():
    parser = argparse.ArgumentParser(
        description="SIFT Feature Extraction Benchmark and Comparison Tool"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/bear01",
        help="Input image or directory (default: data/bear01)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/benchmark",
        help="Output directory (default: output/benchmark)"
    )
    parser.add_argument(
        "--max-features", "-m",
        type=int,
        default=1000,
        help="Maximum features to extract (default: 1000)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations (default: 2)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Timed iterations (default: 5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of images to process (default: 10)"
    )
    parser.add_argument(
        "--skip-colmap",
        action="store_true",
        help="Skip COLMAP benchmark"
    )
    parser.add_argument(
        "--skip-standalone",
        action="store_true",
        help="Skip standalone C++ benchmark"
    )
    parser.add_argument(
        "--skip-wgpu",
        action="store_true",
        help="Skip sift-wgpu benchmark"
    )
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
