//! SIFT Feature Extraction CLI using sift-wgpu with VLFeat-compatible parameters

use clap::Parser;
use sift_wgpu_wrapper::{SiftExtractor, SiftOptions, descriptor_to_bytes, transform_vlfeat_to_ubc};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// SIFT Feature Extractor (sift-wgpu wrapper)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image path
    image_path: PathBuf,
    
    /// Output directory
    output_dir: PathBuf,
    
    /// Maximum number of features (default: 8192)
    #[arg(short, long, default_value = "8192")]
    max_features: usize,
    
    /// First octave (-1 = upsample, default: -1)
    #[arg(long, default_value = "-1")]
    first_octave: i32,
    
    /// Number of octaves (default: 4)
    #[arg(long, default_value = "4")]
    num_octaves: u32,
    
    /// Octave levels/intervals (default: 3)
    #[arg(long, default_value = "3")]
    octave_levels: u32,
    
    /// Peak threshold (default: 0.02/levels)
    #[arg(long)]
    peak_thresh: Option<f32>,
    
    /// Edge threshold (default: 10.0)
    #[arg(long, default_value = "10.0")]
    edge_thresh: f32,
    
    /// Max orientations per keypoint (default: 2)
    #[arg(long, default_value = "2")]
    max_orientations: usize,
    
    /// Fix orientation to 0 (upright features)
    #[arg(long)]
    upright: bool,
}

fn main() {
    let args = Args::parse();
    
    // Build options
    let peak_threshold = args.peak_thresh
        .unwrap_or_else(|| 0.02 / args.octave_levels as f32);
    
    let options = SiftOptions {
        sigma: 1.6,
        num_octaves: args.num_octaves,
        num_intervals: args.octave_levels,
        first_octave: args.first_octave,
        peak_threshold,
        edge_threshold: args.edge_thresh,
        max_features: args.max_features,
        max_orientations: args.max_orientations,
        upright: args.upright,
    };
    
    // Create output directory
    fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");
    
    // Load image
    println!("Loading image: {}", args.image_path.display());
    let img = image::open(&args.image_path).expect("Failed to load image");
    let (width, height) = (img.width(), img.height());
    println!("Image size: {} x {}", width, height);
    
    // Create extractor
    println!("Extracting SIFT features (max {})...", args.max_features);
    let extractor = SiftExtractor::with_options(options);
    
    // Extract features
    let (keypoints, descriptors) = extractor.extract(&img);
    println!("Extracted {} features", keypoints.len());
    
    // Generate output paths
    let stem = args.image_path.file_stem().unwrap().to_str().unwrap();
    let output_txt = args.output_dir.join(format!("{}_keypoints.txt", stem));
    let output_csv = args.output_dir.join(format!("{}_keypoints.csv", stem));
    let output_bin = args.output_dir.join(format!("{}_descriptors.bin", stem));
    
    // Write text format (COLMAP compatible)
    write_keypoints_txt(&output_txt, &keypoints, &descriptors);
    
    // Write CSV format
    write_keypoints_csv(&output_csv, &keypoints);
    
    // Write binary descriptors
    write_descriptors_binary(&output_bin, &descriptors);
    
    // Summary
    println!("\n=== Extraction Summary ===");
    println!("Image: {}", args.image_path.display());
    println!("Size: {} x {}", width, height);
    println!("Features: {}", keypoints.len());
    println!("Descriptor dim: 128");
    println!("Output text: {}", output_txt.display());
    println!("Output CSV: {}", output_csv.display());
    println!("Output binary: {}", output_bin.display());
}

/// Write keypoints in COLMAP text format
fn write_keypoints_txt(path: &PathBuf, keypoints: &[sift::KeyPoint], descriptors: &[Vec<f32>]) {
    let file = File::create(path).expect("Failed to create text file");
    let mut writer = BufWriter::new(file);
    
    // Header: num_features descriptor_dim
    writeln!(writer, "{} 128", keypoints.len()).unwrap();
    
    // Each feature: x y scale orientation d1 d2 ... d128
    for (kp, desc) in keypoints.iter().zip(descriptors.iter()) {
        // Convert to bytes and transform to UBC format
        let desc_bytes = descriptor_to_bytes(desc);
        let desc_ubc = transform_vlfeat_to_ubc(&desc_bytes);
        
        write!(writer, "{:.6} {:.6} {:.6} {:.6}", kp.x, kp.y, kp.size, kp.angle).unwrap();
        for &d in desc_ubc.iter() {
            write!(writer, " {}", d).unwrap();
        }
        writeln!(writer).unwrap();
    }
}

/// Write keypoints in CSV format
fn write_keypoints_csv(path: &PathBuf, keypoints: &[sift::KeyPoint]) {
    let file = File::create(path).expect("Failed to create CSV file");
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "x,y,scale,orientation").unwrap();
    for kp in keypoints {
        writeln!(writer, "{:.6},{:.6},{:.6},{:.6}", kp.x, kp.y, kp.size, kp.angle).unwrap();
    }
}

/// Write descriptors in binary format
fn write_descriptors_binary(path: &PathBuf, descriptors: &[Vec<f32>]) {
    let file = File::create(path).expect("Failed to create binary file");
    let mut writer = BufWriter::new(file);
    
    // Write dimensions
    let num_features = descriptors.len() as i32;
    let descriptor_dim = 128i32;
    writer.write_all(&num_features.to_le_bytes()).unwrap();
    writer.write_all(&descriptor_dim.to_le_bytes()).unwrap();
    
    // Write descriptors (converted to bytes)
    for desc in descriptors {
        let desc_bytes = descriptor_to_bytes(desc);
        writer.write_all(&desc_bytes).unwrap();
    }
}
