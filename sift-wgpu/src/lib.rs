//! Minimal wrapper around sift-wgpu with VLFeat-compatible parameters
//!
//! This wrapper adds the following features missing from sift-wgpu:
//! - `first_octave` parameter (with 2x upsampling when -1)
//! - `max_features` limiting (keeps largest scale features)
//! - `max_orientations` limiting per keypoint
//! - Proper peak_threshold scaling

use image::{DynamicImage, imageops::FilterType};

// Re-export sift-wgpu types (the lib name is "sift")
pub use sift::{KeyPoint, Sift, SiftBackend};

/// SIFT extraction options matching VLFeat/COLMAP behavior
#[derive(Debug, Clone)]
pub struct SiftOptions {
    /// Base sigma for Gaussian blur (default: 1.6)
    pub sigma: f32,
    /// Number of octaves (default: 4)
    pub num_octaves: u32,
    /// Number of intervals per octave (default: 3)
    pub num_intervals: u32,
    /// First octave index. -1 means upsample image 2x first (default: -1)
    pub first_octave: i32,
    /// Peak threshold for keypoint detection (default: 0.02 / num_intervals)
    pub peak_threshold: f32,
    /// Edge threshold (default: 10.0)
    pub edge_threshold: f32,
    /// Maximum number of features to keep (default: 8192)
    pub max_features: usize,
    /// Maximum orientations per keypoint (default: 2)
    pub max_orientations: usize,
    /// Fix orientation to 0 for upright features (default: false)
    pub upright: bool,
}

impl Default for SiftOptions {
    fn default() -> Self {
        let num_intervals = 3;
        Self {
            sigma: 1.6,
            num_octaves: 4,
            num_intervals,
            first_octave: -1,
            peak_threshold: 0.02 / num_intervals as f32,
            edge_threshold: 10.0,
            max_features: 8192,
            max_orientations: 2,
            upright: false,
        }
    }
}

/// SIFT extractor wrapping sift-wgpu with VLFeat-compatible behavior
pub struct SiftExtractor {
    options: SiftOptions,
}

impl SiftExtractor {
    /// Create a new SIFT extractor with default options
    pub fn new() -> Self {
        Self::with_options(SiftOptions::default())
    }
    
    /// Create a new SIFT extractor with custom options
    pub fn with_options(options: SiftOptions) -> Self {
        Self { options }
    }
    
    /// Extract keypoints and descriptors from an image
    pub fn extract(&self, image: &DynamicImage) -> (Vec<KeyPoint>, Vec<Vec<f32>>) {
        // Handle first_octave by resizing the image
        let (processed_image, octave_offset) = if self.options.first_octave == -1 {
            // Upsample 2x
            let (w, h) = (image.width() * 2, image.height() * 2);
            let resized = image.resize_exact(w, h, FilterType::Lanczos3);
            (resized, 0)
        } else if self.options.first_octave > 0 {
            // Downsample
            let scale = 2.0_f32.powi(self.options.first_octave);
            let w = (image.width() as f32 / scale) as u32;
            let h = (image.height() as f32 / scale) as u32;
            let resized = image.resize_exact(w, h, FilterType::Lanczos3);
            (resized, 0)
        } else {
            (image.clone(), 0)
        };
        
        // Create sift-wgpu extractor with adjusted parameters
        let sift = Sift::new(
            self.options.sigma,
            self.options.num_octaves,
            self.options.num_intervals,
            0.5, // assumed_blur
            self.options.peak_threshold * self.options.num_intervals as f32, // sift-wgpu divides by num_intervals
            self.options.edge_threshold,
        );
        
        // Extract features using sift-wgpu
        let (mut keypoints, descriptors) = sift.detect_and_compute(&processed_image);
        
        // Adjust keypoint coordinates if we upsampled
        if self.options.first_octave == -1 {
            for kp in &mut keypoints {
                kp.x /= 2.0;
                kp.y /= 2.0;
                kp.size /= 2.0;
            }
        } else if self.options.first_octave > 0 {
            let scale = 2.0_f32.powi(self.options.first_octave);
            for kp in &mut keypoints {
                kp.x *= scale;
                kp.y *= scale;
                kp.size *= scale;
            }
        }
        
        // Limit orientations per keypoint
        let (keypoints, descriptors) = self.limit_orientations(keypoints, descriptors);
        
        // Limit total features
        let (keypoints, descriptors) = self.limit_features(keypoints, descriptors);
        
        (keypoints, descriptors)
    }
    
    /// Limit orientations per keypoint by keeping only the strongest orientations
    fn limit_orientations(&self, keypoints: Vec<KeyPoint>, descriptors: Vec<Vec<f32>>) -> (Vec<KeyPoint>, Vec<Vec<f32>>) {
        if self.options.max_orientations >= 4 { // sift-wgpu max is 4
            return (keypoints, descriptors);
        }
        
        // Group keypoints by approximate position
        // Since sift-wgpu already limits to 4 orientations, we just need to handle
        // cases where max_orientations < 4
        let mut result_kps = Vec::new();
        let mut result_descs = Vec::new();
        
        // Sort by position and response to group orientations
        let mut indexed: Vec<_> = keypoints.into_iter().zip(descriptors.into_iter()).enumerate().collect();
        indexed.sort_by(|a, b| {
            // Sort by (rounded_x, rounded_y, -response) to group same-location keypoints
            let ax = (a.1 .0.x * 10.0).round() as i32;
            let ay = (a.1 .0.y * 10.0).round() as i32;
            let bx = (b.1 .0.x * 10.0).round() as i32;
            let by = (b.1 .0.y * 10.0).round() as i32;
            
            match ax.cmp(&bx) {
                std::cmp::Ordering::Equal => match ay.cmp(&by) {
                    std::cmp::Ordering::Equal => {
                        // Sort by response (descending) to keep strongest orientations
                        b.1 .0.response.partial_cmp(&a.1 .0.response).unwrap()
                    }
                    other => other,
                },
                other => other,
            }
        });
        
        // Track position groups and limit orientations
        let mut prev_x = i32::MIN;
        let mut prev_y = i32::MIN;
        let mut count_in_group = 0;
        
        for (_, (kp, desc)) in indexed {
            let curr_x = (kp.x * 10.0).round() as i32;
            let curr_y = (kp.y * 10.0).round() as i32;
            
            if curr_x != prev_x || curr_y != prev_y {
                // New position group
                prev_x = curr_x;
                prev_y = curr_y;
                count_in_group = 1;
                result_kps.push(kp);
                result_descs.push(desc);
            } else if count_in_group < self.options.max_orientations {
                // Same position, but under limit
                count_in_group += 1;
                result_kps.push(kp);
                result_descs.push(desc);
            }
            // Otherwise, skip this orientation
        }
        
        (result_kps, result_descs)
    }
    
    /// Limit features to max_features, keeping largest scale features
    fn limit_features(&self, keypoints: Vec<KeyPoint>, descriptors: Vec<Vec<f32>>) -> (Vec<KeyPoint>, Vec<Vec<f32>>) {
        if keypoints.len() <= self.options.max_features {
            return (keypoints, descriptors);
        }
        
        // Sort by size (scale) descending and keep largest
        let mut indexed: Vec<_> = keypoints.into_iter().zip(descriptors.into_iter()).enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1 .0.size.partial_cmp(&a.1 .0.size).unwrap()
        });
        indexed.truncate(self.options.max_features);
        
        let (final_kps, final_descs): (Vec<_>, Vec<_>) = indexed
            .into_iter()
            .map(|(_, (kp, desc))| (kp, desc))
            .unzip();
        
        (final_kps, final_descs)
    }
}

impl Default for SiftExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert float descriptor to unsigned byte format
pub fn descriptor_to_bytes(desc: &[f32]) -> Vec<u8> {
    desc.iter().map(|&v| {
        let scaled = (v.abs() * 512.0).round() as f32;
        scaled.clamp(0.0, 255.0) as u8
    }).collect()
}

/// Transform VLFeat descriptor format to UBC format
pub fn transform_vlfeat_to_ubc(desc: &[u8]) -> [u8; 128] {
    let q = [0, 7, 6, 5, 4, 3, 2, 1];
    let mut result = [0u8; 128];
    
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..8 {
                let src_idx = 8 * (j + 4 * i) + k;
                let dst_idx = 8 * (j + 4 * i) + q[k];
                if src_idx < desc.len() {
                    result[dst_idx] = desc[src_idx];
                }
            }
        }
    }
    
    result
}
