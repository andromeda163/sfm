//! SIFT Feature Extraction Benchmark
//!
//! Compares runtime performance of different SIFT implementations

use anyhow::Result;
use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageReader};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::collections::HashMap;

/// SIFT benchmark tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image or directory
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "output/benchmark")]
    output: PathBuf,

    /// Maximum number of features to extract
    #[arg(short, long, default_value = "1000")]
    max_features: usize,

    /// Number of warmup iterations (not timed)
    #[arg(long, default_value = "2")]
    warmup: usize,

    /// Number of timed iterations
    #[arg(long, default_value = "5")]
    iterations: usize,

    /// Limit number of images to process
    #[arg(long, default_value = "0")]
    limit: usize,

    /// Output format (json, csv, text)
    #[arg(long, default_value = "text")]
    format: String,

    /// Run only sift-wgpu benchmark
    #[arg(long)]
    wgpu_only: bool,
}

/// Benchmark results for a single image
#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResult {
    image: String,
    width: u32,
    height: u32,
    implementation: String,
    num_features: usize,
    total_time_ms: f64,
    avg_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    throughput_mpixels_per_sec: f64,
    features_per_sec: f64,
}

/// Summary statistics across all images
#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkSummary {
    implementation: String,
    total_images: usize,
    total_features: usize,
    total_time_ms: f64,
    avg_time_per_image_ms: f64,
    avg_features_per_image: f64,
    total_mpixels: f64,
    avg_throughput_mpixels_per_sec: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Create output directory
    std::fs::create_dir_all(&args.output)?;
    
    // Collect images to process
    let mut images = collect_images(&args.input)?;
    
    // Apply limit if specified
    if args.limit > 0 && images.len() > args.limit {
        images.truncate(args.limit);
    }
    
    if images.is_empty() {
        eprintln!("No images found in {}", args.input.display());
        return Ok(());
    }
    
    println!("SIFT Benchmark");
    println!("==============");
    println!("Images: {}", images.len());
    println!("Max features: {}", args.max_features);
    println!("Warmup iterations: {}", args.warmup);
    println!("Timed iterations: {}", args.iterations);
    println!();
    
    // Run benchmarks
    let mut all_results: Vec<BenchmarkResult> = Vec::new();
    
    // Benchmark sift-wgpu
    println!("Running sift-wgpu benchmark...");
    let wgpu_results = benchmark_sift_wgpu(&images, &args)?;
    all_results.extend(wgpu_results);
    
    // Print results
    println!();
    print_results(&all_results, &args.format);
    
    // Write results to file
    let output_file = args.output.join("benchmark_results.json");
    let json = serde_json::to_string_pretty(&all_results)?;
    std::fs::write(&output_file, json)?;
    println!("\nResults written to: {}", output_file.display());
    
    // Print summary
    print_summary(&all_results);
    
    Ok(())
}

/// Collect all images from a path (file or directory)
fn collect_images(path: &Path) -> Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    
    if path.is_file() {
        if is_image_file(path) {
            images.push(path.to_path_buf());
        }
    } else if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && is_image_file(&path) {
                images.push(path);
            }
        }
        images.sort();
    }
    
    Ok(images)
}

/// Check if a file is an image
fn is_image_file(path: &Path) -> bool {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    
    matches!(ext.as_deref(), Some("jpg") | Some("jpeg") | Some("png") | Some("bmp") | Some("tiff"))
}

/// Load image and convert to grayscale f32
fn load_image_grayscale(path: &Path) -> Result<(DynamicImage, u32, u32, Vec<f32>)> {
    let img = ImageReader::open(path)?.decode()?;
    let (width, height) = (img.width(), img.height());
    
    // Convert to grayscale
    let gray = img.to_luma8();
    
    // Convert to f32 in [0, 1]
    let data: Vec<f32> = gray.iter()
        .map(|&p| p as f32 / 255.0)
        .collect();
    
    Ok((img, width, height, data))
}

/// Benchmark sift-wgpu implementation
fn benchmark_sift_wgpu(images: &[PathBuf], args: &Args) -> Result<Vec<BenchmarkResult>> {
    let mut results = Vec::new();
    
    // Import from sift-wgpu crate (library name is 'sift')
    use sift::{Sift, SiftBackend};
    
    for image_path in images {
        let img_name = image_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Load image using sift's loader
        let img = sift::load_image_dyn(image_path.to_str().unwrap_or(""))?;
        let (width, height) = (img.width(), img.height());
        let mpixels = (width * height) as f64 / 1_000_000.0;
        
        // Create SIFT extractor with CPU backend
        // Sift::new(sigma, num_octaves, num_intervals, assumed_blur, contrast_threshold, edge_threshold)
        let sift = Sift::new(
            1.6,     // sigma
            4,       // num_octaves
            3,       // num_intervals
            0.5,     // assumed_blur
            0.04,    // contrast_threshold
            10.0,    // edge_threshold
        );
        
        // Warmup runs
        for _ in 0..args.warmup {
            let _ = sift.detect_and_compute(&img);
        }
        
        // Timed runs
        let mut times = Vec::with_capacity(args.iterations);
        let mut num_features = 0;
        
        for _ in 0..args.iterations {
            let start = Instant::now();
            let (kps, _descs) = sift.detect_and_compute(&img);
            let elapsed = start.elapsed();
            
            num_features = kps.len();
            times.push(elapsed.as_secs_f64() * 1000.0);
        }
        
        let total_time: f64 = times.iter().sum();
        let avg_time = total_time / times.len() as f64;
        let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        results.push(BenchmarkResult {
            image: img_name,
            width,
            height,
            implementation: "sift-wgpu".to_string(),
            num_features,
            total_time_ms: total_time,
            avg_time_ms: avg_time,
            min_time_ms: min_time,
            max_time_ms: max_time,
            throughput_mpixels_per_sec: mpixels / (avg_time / 1000.0),
            features_per_sec: num_features as f64 / (avg_time / 1000.0),
        });
        
        println!("  {} ({}x{}): {:.2}ms, {} features", 
                 results.last().unwrap().image,
                 width, height,
                 avg_time,
                 num_features);
    }
    
    Ok(results)
}

/// Print results in specified format
fn print_results(results: &[BenchmarkResult], format: &str) {
    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(results).unwrap());
        }
        "csv" => {
            println!("image,width,height,implementation,num_features,total_time_ms,avg_time_ms,min_time_ms,max_time_ms,throughput_mpixels_per_sec");
            for r in results {
                println!("{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2}",
                    r.image, r.width, r.height, r.implementation, r.num_features,
                    r.total_time_ms, r.avg_time_ms, r.min_time_ms, r.max_time_ms,
                    r.throughput_mpixels_per_sec);
            }
        }
        _ => {
            // Text format
            println!("{:<25} {:>8} {:>10} {:>12} {:>10} {:>12}",
                "Image", "Size", "Features", "Avg Time", "Min Time", "Throughput");
            println!("{}", "-".repeat(80));
            for r in results {
                println!("{:<25} {:>4}x{:<4} {:>10} {:>10.2}ms {:>10.2}ms {:>10.2} MP/s",
                    r.image,
                    r.width, r.height,
                    r.num_features,
                    r.avg_time_ms,
                    r.min_time_ms,
                    r.throughput_mpixels_per_sec);
            }
        }
    }
}

/// Print summary statistics
fn print_summary(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(80));
    println!("SUMMARY");
    println!("{}", "=".repeat(80));
    
    // Group by implementation
    let mut by_impl: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for r in results {
        by_impl.entry(r.implementation.clone())
            .or_default()
            .push(r);
    }
    
    println!("{:<15} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "Implementation", "Images", "Total Time", "Avg Time", "Avg Features", "Throughput");
    println!("{}", "-".repeat(80));
    
    for (impl_name, impl_results) in &by_impl {
        let total_images = impl_results.len();
        let total_time: f64 = impl_results.iter().map(|r| r.total_time_ms).sum();
        let avg_time = total_time / total_images as f64;
        let avg_features: f64 = impl_results.iter()
            .map(|r| r.num_features as f64)
            .sum::<f64>() / total_images as f64;
        let total_mpixels: f64 = impl_results.iter()
            .map(|r| (r.width * r.height) as f64 / 1_000_000.0)
            .sum();
        let throughput = total_mpixels / (total_time / 1000.0);
        
        println!("{:<15} {:>8} {:>10.2}ms {:>10.2}ms {:>12.1} {:>10.2} MP/s",
            impl_name, total_images, total_time, avg_time, avg_features, throughput);
    }
}
