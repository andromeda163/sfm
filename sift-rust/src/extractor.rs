//! SIFT Feature Extractor
//!
//! Implements the SIFT algorithm matching COLMAP/VLFeat behavior

use crate::keypoint::{Keypoint, Keypoints};
use crate::descriptor::{self, Descriptor, Normalization};
use crate::pyramid::{
    build_gaussian_pyramid, build_dog_pyramid, upsample_image, resize_image,
    get_pixel_clamped, get_pixel_bilinear, GaussianPyramid, DogPyramid,
};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;
use rayon::prelude::*;
use std::f32::consts::PI;

/// SIFT extraction options
#[derive(Debug, Clone)]
pub struct SiftOptions {
    /// Base sigma for Gaussian blur (default: 1.6)
    pub sigma: f32,
    /// Number of octaves in the pyramid (default: 4)
    pub num_octaves: u32,
    /// Number of intervals per octave (default: 3)
    pub num_intervals: u32,
    /// First octave index. -1 means upsample image 2x first (default: -1)
    pub first_octave: i32,
    /// Assumed blur of input image (default: 0.5)
    pub assumed_blur: f32,
    /// Peak threshold for keypoint detection (default: 0.02 / num_intervals)
    pub peak_threshold: f32,
    /// Edge threshold for rejecting edge-like keypoints (default: 10.0)
    pub edge_threshold: f32,
    /// Maximum number of features to keep (default: 8192)
    pub max_features: usize,
    /// Maximum orientations per keypoint (default: 2)
    pub max_orientations: usize,
    /// Fix orientation to 0 for upright features (default: false)
    pub upright: bool,
    /// Descriptor normalization method (default: L1Root)
    pub normalization: Normalization,
}

impl Default for SiftOptions {
    fn default() -> Self {
        let num_intervals = 3;
        Self {
            sigma: 1.6,
            num_octaves: 4,
            num_intervals,
            first_octave: -1,
            assumed_blur: 0.5,
            peak_threshold: 0.02 / num_intervals as f32,
            edge_threshold: 10.0,
            max_features: 8192,
            max_orientations: 2,
            upright: false,
            normalization: Normalization::L1Root,
        }
    }
}

impl SiftOptions {
    /// Create options with custom peak threshold
    pub fn with_peak_threshold(mut self, threshold: f32) -> Self {
        self.peak_threshold = threshold;
        self
    }
    
    /// Create options with custom first octave
    pub fn with_first_octave(mut self, octave: i32) -> Self {
        self.first_octave = octave;
        self
    }
    
    /// Create options with custom max features
    pub fn with_max_features(mut self, max: usize) -> Self {
        self.max_features = max;
        self
    }
    
    /// Create options with custom max orientations
    pub fn with_max_orientations(mut self, max: usize) -> Self {
        self.max_orientations = max;
        self
    }
    
    /// Validate options
    pub fn validate(&self) -> bool {
        self.sigma > 0.0
            && self.num_octaves > 0
            && self.num_intervals > 0
            && self.peak_threshold > 0.0
            && self.edge_threshold > 0.0
    }
}

/// SIFT feature extractor
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
        assert!(options.validate(), "Invalid SIFT options");
        Self { options }
    }
    
    /// Extract keypoints and descriptors from an image
    pub fn extract(&self, image: &DynamicImage) -> (Keypoints, Vec<Descriptor>) {
        // Convert to grayscale
        let gray = image.to_luma8();
        
        // Prepare base image (handle first_octave)
        let base_image = self.prepare_base_image(&gray);
        
        // Build pyramids
        let gaussian_pyramid = build_gaussian_pyramid(
            &base_image,
            self.options.num_octaves as usize,
            self.options.num_intervals,
            self.options.sigma,
        );
        let dog_pyramid = build_dog_pyramid(&gaussian_pyramid);
        
        // Find keypoints
        let keypoints = self.detect_keypoints(&dog_pyramid);
        
        // Assign orientations
        let oriented_keypoints = self.assign_orientations(&keypoints, &gaussian_pyramid);
        
        // Compute descriptors
        let descriptors = self.compute_descriptors(&oriented_keypoints, &gaussian_pyramid);
        
        // Limit features if needed
        let (final_keypoints, final_descriptors) = self.limit_features(oriented_keypoints, descriptors);
        
        (final_keypoints, final_descriptors)
    }
    
    /// Prepare the base image, handling first_octave parameter
    fn prepare_base_image(&self, gray: &GrayImage) -> GrayImage {
        // Apply initial blur to reach sigma
        let initial_blur = if self.options.sigma > self.options.assumed_blur {
            (self.options.sigma.powi(2) - self.options.assumed_blur.powi(2)).sqrt()
        } else {
            0.0
        };
        
        let blurred = if initial_blur > 1e-4 {
            gaussian_blur_f32(gray, initial_blur)
        } else {
            gray.clone()
        };
        
        // Handle first_octave
        if self.options.first_octave == -1 {
            // Upsample image 2x
            upsample_image(&blurred)
        } else if self.options.first_octave > 0 {
            // Downsample by 2^first_octave
            let scale = 2.0_f32.powi(self.options.first_octave);
            let new_width = (blurred.width() as f32 / scale) as u32;
            let new_height = (blurred.height() as f32 / scale) as u32;
            resize_image(&blurred, new_width, new_height)
        } else {
            blurred
        }
    }
    
    /// Detect keypoints in the DoG pyramid
    fn detect_keypoints(&self, dog_pyramid: &DogPyramid) -> Keypoints {
        let border = 5i32;
        let num_intervals = self.options.num_intervals as usize;
        
        // The effective octave offset due to first_octave
        // When first_octave=-1, the image is upsampled 2x, so octave 0 is actually at 2x scale
        let octave_offset = self.options.first_octave;
        
        // Find initial extrema in parallel
        let initial_keypoints: Vec<Keypoint> = dog_pyramid
            .par_iter()
            .enumerate()
            .flat_map(|(octave_idx, dog_octave)| {
                if dog_octave.is_empty() {
                    return Vec::new();
                }
                
                let (width, height) = dog_octave[0].dimensions();
                let width_i32 = width as i32;
                let height_i32 = height as i32;
                
                // Scale factor to convert from octave coordinates to original image coordinates
                // For first_octave=-1, octave 0 is at 2x scale, so divide by 2
                let scale_factor = 2.0_f32.powi(octave_idx as i32 + octave_offset);
                
                // Process each scale in the octave
                (1..=num_intervals)
                    .into_par_iter()
                    .filter(|&s_idx| s_idx < dog_octave.len() - 1)
                    .flat_map(|s_idx| {
                        let img_prev = &dog_octave[s_idx - 1];
                        let img_curr = &dog_octave[s_idx];
                        let img_next = &dog_octave[s_idx + 1];
                        
                        // Process each pixel
                        (border..(height_i32 - border))
                            .into_par_iter()
                            .flat_map(|y| {
                                let mut row_kps = Vec::new();
                                for x in border..(width_i32 - border) {
                                    let val = get_pixel_clamped(img_curr, x, y);
                                    
                                    // Check if local extremum
                                    let (is_max, is_min) = check_extremum(
                                        val, img_prev, img_curr, img_next, x, y
                                    );
                                    
                                    if is_max || is_min {
                                        row_kps.push(Keypoint {
                                            x: (x as f32 + 0.5) * scale_factor,
                                            y: (y as f32 + 0.5) * scale_factor,
                                            scale: 0.0, // Will be computed during refinement
                                            orientation: 0.0,
                                            octave: octave_idx as i32,
                                            layer: s_idx as i32,
                                            response: val,
                                        });
                                    }
                                }
                                row_kps
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        
        // Refine and filter keypoints
        self.refine_keypoints(initial_keypoints, dog_pyramid)
    }
    
    /// Refine keypoint locations and filter by contrast/edge response
    fn refine_keypoints(&self, keypoints: Keypoints, dog_pyramid: &DogPyramid) -> Keypoints {
        let octave_offset = self.options.first_octave;
        
        keypoints
            .par_iter()
            .filter_map(|kp| {
                let octave_idx = kp.octave as usize;
                let layer_idx = kp.layer as usize;
                let dog_octave = &dog_pyramid[octave_idx];
                
                if layer_idx < 1 || layer_idx >= dog_octave.len() - 1 {
                    return None;
                }
                
                // Scale factor: converts from original image coords to octave coords
                // For first_octave=-1, octave 0 is at 2x scale
                let scale_factor = 2.0_f32.powi(kp.octave + octave_offset);
                let mut x = kp.x / scale_factor;
                let mut y = kp.y / scale_factor;
                let mut layer = layer_idx as f32;
                
                // Iterative refinement
                for _ in 0..5 {
                    let layer_int = layer.round() as usize;
                    if layer_int < 1 || layer_int >= dog_octave.len() - 1 {
                        return None;
                    }
                    
                    let img_prev = &dog_octave[layer_int - 1];
                    let img_curr = &dog_octave[layer_int];
                    let img_next = &dog_octave[layer_int + 1];
                    
                    let (width, height) = img_curr.dimensions();
                    // Use truncation (as i32) instead of rounding to match sift-wgpu
                    let x_int = x as i32;
                    let y_int = y as i32;
                    
                    if x_int < 1 || x_int >= width as i32 - 1
                        || y_int < 1 || y_int >= height as i32 - 1 {
                        return None;
                    }
                    
                    // Compute gradient and Hessian
                    let dx = (get_pixel_clamped(img_curr, x_int + 1, y_int)
                        - get_pixel_clamped(img_curr, x_int - 1, y_int)) / 2.0;
                    let dy = (get_pixel_clamped(img_curr, x_int, y_int + 1)
                        - get_pixel_clamped(img_curr, x_int, y_int - 1)) / 2.0;
                    let ds = (get_pixel_clamped(img_next, x_int, y_int)
                        - get_pixel_clamped(img_prev, x_int, y_int)) / 2.0;
                    
                    let center = get_pixel_clamped(img_curr, x_int, y_int);
                    let dxx = get_pixel_clamped(img_curr, x_int + 1, y_int)
                        + get_pixel_clamped(img_curr, x_int - 1, y_int)
                        - 2.0 * center;
                    let dyy = get_pixel_clamped(img_curr, x_int, y_int + 1)
                        + get_pixel_clamped(img_curr, x_int, y_int - 1)
                        - 2.0 * center;
                    let dss = get_pixel_clamped(img_next, x_int, y_int)
                        + get_pixel_clamped(img_prev, x_int, y_int)
                        - 2.0 * center;
                    let dxy = (get_pixel_clamped(img_curr, x_int + 1, y_int + 1)
                        - get_pixel_clamped(img_curr, x_int - 1, y_int + 1)
                        - get_pixel_clamped(img_curr, x_int + 1, y_int - 1)
                        + get_pixel_clamped(img_curr, x_int - 1, y_int - 1)) / 4.0;
                    let dxs = (get_pixel_clamped(img_next, x_int + 1, y_int)
                        - get_pixel_clamped(img_next, x_int - 1, y_int)
                        - get_pixel_clamped(img_prev, x_int + 1, y_int)
                        + get_pixel_clamped(img_prev, x_int - 1, y_int)) / 4.0;
                    let dys = (get_pixel_clamped(img_next, x_int, y_int + 1)
                        - get_pixel_clamped(img_next, x_int, y_int - 1)
                        - get_pixel_clamped(img_prev, x_int, y_int + 1)
                        + get_pixel_clamped(img_prev, x_int, y_int - 1)) / 4.0;
                    
                    // Solve for offset using Cramer's rule
                    let hessian = [
                        [dxx, dxy, dxs],
                        [dxy, dyy, dys],
                        [dxs, dys, dss],
                    ];
                    let gradient = [-dx, -dy, -ds];
                    
                    if let Some(offset) = solve_3x3(hessian, gradient) {
                        if offset[0].abs() < 0.5 && offset[1].abs() < 0.5 && offset[2].abs() < 0.5 {
                            // Converged - check contrast
                            let interpolated_val = center + 0.5 * (dx * offset[0] + dy * offset[1] + ds * offset[2]);
                            
                            if interpolated_val.abs() < self.options.peak_threshold {
                                return None;
                            }
                            
                            // Check edge response
                            let det = dxx * dyy - dxy * dxy;
                            let trace = dxx + dyy;
                            
                            if det <= 0.0 {
                                return None;
                            }
                            
                            let edge_ratio = trace * trace / det;
                            let edge_threshold_sq = (self.options.edge_threshold + 1.0).powi(2) / self.options.edge_threshold;
                            
                            if edge_ratio >= edge_threshold_sq {
                                return None;
                            }
                            
                            // Compute final scale
                            // Scale = sigma * 2^(octave + layer/num_intervals)
                            // Where octave is relative to original image (accounting for first_octave)
                            let final_layer = layer_int as f32 + offset[2];
                            let effective_octave = kp.octave as f32 + octave_offset as f32;
                            let scale = self.options.sigma
                                * 2.0_f32.powf(effective_octave + final_layer / self.options.num_intervals as f32);
                            
                            return Some(Keypoint {
                                x: (x_int as f32 + offset[0]) * scale_factor,
                                y: (y_int as f32 + offset[1]) * scale_factor,
                                scale,
                                orientation: 0.0,
                                octave: kp.octave,
                                layer: final_layer.round() as i32,
                                response: interpolated_val,
                            });
                        } else {
                            // Update position and continue
                            x = x_int as f32 + offset[0];
                            y = y_int as f32 + offset[1];
                            layer = layer_int as f32 + offset[2];
                        }
                    } else {
                        return None;
                    }
                }
                
                None
            })
            .collect()
    }
    
    /// Assign orientations to keypoints
    fn assign_orientations(&self, keypoints: &Keypoints, gaussian_pyramid: &GaussianPyramid) -> Keypoints {
        let octave_offset = self.options.first_octave;
        
        keypoints
            .par_iter()
            .flat_map(|kp| {
                if self.options.upright {
                    let mut kp = *kp;
                    kp.orientation = 0.0;
                    return vec![kp];
                }
                
                let octave_idx = kp.octave as usize;
                let layer_idx = kp.layer.clamp(0, self.options.num_intervals as i32 + 2) as usize;
                
                if octave_idx >= gaussian_pyramid.len()
                    || layer_idx >= gaussian_pyramid[octave_idx].len() {
                    return vec![];
                }
                
                let gauss_img = &gaussian_pyramid[octave_idx][layer_idx];
                let (width, height) = gauss_img.dimensions();
                let scale_factor = 2.0_f32.powi(kp.octave + octave_offset);
                let x_octave = kp.x / scale_factor;
                let y_octave = kp.y / scale_factor;
                let sigma_octave = kp.scale / scale_factor;
                
                if sigma_octave <= 0.0 {
                    return vec![];
                }
                
                // Build orientation histogram
                let radius = (3.0 * 1.5 * sigma_octave).round() as i32;
                let weight_sigma = 1.5 * sigma_octave;
                let weight_denom = 2.0 * weight_sigma * weight_sigma;
                let mut hist = [0.0f32; 36];
                
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let x_img = (x_octave + dx as f32).round() as i32;
                        let y_img = (y_octave + dy as f32).round() as i32;
                        
                        if x_img < 1 || x_img >= width as i32 - 1
                            || y_img < 1 || y_img >= height as i32 - 1 {
                            continue;
                        }
                        
                        let grad_x = get_pixel_bilinear(gauss_img, x_img as f32 + 1.0, y_img as f32)
                            - get_pixel_bilinear(gauss_img, x_img as f32 - 1.0, y_img as f32);
                        let grad_y = get_pixel_bilinear(gauss_img, x_img as f32, y_img as f32 + 1.0)
                            - get_pixel_bilinear(gauss_img, x_img as f32, y_img as f32 - 1.0);
                        
                        let magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();
                        let angle = grad_y.atan2(grad_x);
                        let angle = if angle < 0.0 { angle + 2.0 * PI } else { angle };
                        
                        let weight = (-(dx as f32 * dx as f32 + dy as f32 * dy as f32) / weight_denom).exp();
                        let bin = ((angle / (2.0 * PI)) * 36.0) as usize % 36;
                        hist[bin] += magnitude * weight;
                    }
                }
                
                // Smooth histogram
                for _ in 0..2 {
                    let prev = hist.clone();
                    for i in 0..36 {
                        hist[i] = (prev[(i + 35) % 36] + prev[i] + prev[(i + 1) % 36]) / 3.0;
                    }
                }
                
                // Find peaks
                let max_val = hist.iter().cloned().fold(0.0, f32::max);
                let threshold = max_val * 0.8;
                
                let mut orientations = Vec::new();
                for i in 0..36 {
                    if hist[i] >= threshold && hist[i] > hist[(i + 35) % 36] && hist[i] > hist[(i + 1) % 36] {
                        // Parabolic interpolation
                        let prev = hist[(i + 35) % 36];
                        let next = hist[(i + 1) % 36];
                        let offset = if (prev - 2.0 * hist[i] + next).abs() > 1e-5 {
                            0.5 * (prev - next) / (prev - 2.0 * hist[i] + next)
                        } else {
                            0.0
                        };
                        
                        let angle = ((i as f32 + 0.5 + offset) * 2.0 * PI / 36.0) % (2.0 * PI);
                        let angle = if angle > PI { angle - 2.0 * PI } else { angle };
                        
                        let mut new_kp = *kp;
                        new_kp.orientation = angle;
                        orientations.push((new_kp, hist[i]));
                    }
                }
                
                // Sort by histogram value and limit orientations
                orientations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                orientations.truncate(self.options.max_orientations);
                
                orientations.into_iter().map(|(kp, _)| kp).collect()
            })
            .collect()
    }
    
    /// Compute descriptors for keypoints
    fn compute_descriptors(&self, keypoints: &Keypoints, gaussian_pyramid: &GaussianPyramid) -> Vec<Descriptor> {
        let octave_offset = self.options.first_octave;
        
        keypoints
            .par_iter()
            .map(|kp| {
                let octave_idx = kp.octave as usize;
                let layer_idx = kp.layer.clamp(0, self.options.num_intervals as i32 + 2) as usize;
                
                if octave_idx >= gaussian_pyramid.len()
                    || layer_idx >= gaussian_pyramid[octave_idx].len() {
                    return [0u8; 128];
                }
                
                let gauss_img = &gaussian_pyramid[octave_idx][layer_idx];
                let (width, height) = gauss_img.dimensions();
                let scale_factor = 2.0_f32.powi(kp.octave + octave_offset);
                let x_octave = kp.x / scale_factor;
                let y_octave = kp.y / scale_factor;
                let sigma_octave = kp.scale / scale_factor;
                
                if sigma_octave <= 0.0 {
                    return [0u8; 128];
                }
                
                let angle = kp.orientation;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                
                let bin_width = 3.0 * sigma_octave;
                let window_width = bin_width * 4.0;
                let weight_sigma = 0.5 * window_width;
                let weight_denom = 2.0 * weight_sigma * weight_sigma;
                let sample_radius = (window_width * 0.5 * 2.0_f32.sqrt()).ceil() as i32;
                
                let mut hist = [0.0f32; 128];
                
                for dy in -sample_radius..=sample_radius {
                    for dx in -sample_radius..=sample_radius {
                        let rx = cos_a * dx as f32 + sin_a * dy as f32;
                        let ry = -sin_a * dx as f32 + cos_a * dy as f32;
                        
                        let x_bin = rx / bin_width + 1.5;
                        let y_bin = ry / bin_width + 1.5;
                        
                        if x_bin < -1.0 || x_bin > 4.0 || y_bin < -1.0 || y_bin > 4.0 {
                            continue;
                        }
                        
                        let x_sample = x_octave + dx as f32;
                        let y_sample = y_octave + dy as f32;
                        
                        if x_sample < 1.0 || x_sample >= (width - 1) as f32
                            || y_sample < 1.0 || y_sample >= (height - 1) as f32 {
                            continue;
                        }
                        
                        let grad_x = get_pixel_bilinear(gauss_img, x_sample + 1.0, y_sample)
                            - get_pixel_bilinear(gauss_img, x_sample - 1.0, y_sample);
                        let grad_y = get_pixel_bilinear(gauss_img, x_sample, y_sample + 1.0)
                            - get_pixel_bilinear(gauss_img, x_sample, y_sample - 1.0);
                        
                        let magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();
                        let mut angle_bin = (grad_y.atan2(grad_x) - angle) / (2.0 * PI) * 8.0;
                        
                        // Normalize angle to [0, 8)
                        while angle_bin < 0.0 { angle_bin += 8.0; }
                        while angle_bin >= 8.0 { angle_bin -= 8.0; }
                        
                        let weight = (-(dx as f32 * dx as f32 + dy as f32 * dy as f32) / weight_denom).exp();
                        
                        // Trilinear interpolation into histogram
                        trilinear_interpolate(&mut hist, x_bin, y_bin, angle_bin, magnitude * weight);
                    }
                }
                
                // Normalize descriptor
                match self.options.normalization {
                    Normalization::L2 => descriptor::l2_normalize(&mut hist),
                    Normalization::L1Root => descriptor::l1_root_normalize(&mut hist),
                }
                
                // Convert to bytes
                let desc = descriptor::to_unsigned_byte(&hist);
                
                // Transform to UBC format
                descriptor::transform_vlfeat_to_ubc(&desc)
            })
            .collect()
    }
    
    /// Limit features to max_features, keeping largest scale features
    fn limit_features(&self, keypoints: Keypoints, descriptors: Vec<Descriptor>) -> (Keypoints, Vec<Descriptor>) {
        if keypoints.len() <= self.options.max_features {
            return (keypoints, descriptors);
        }
        
        // Collect with indices and sort by scale (descending)
        let mut indexed: Vec<_> = keypoints.into_iter().zip(descriptors.into_iter()).enumerate().collect();
        indexed.sort_by(|a, b| b.1 .0.scale.partial_cmp(&a.1 .0.scale).unwrap());
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

/// Check if a pixel is a local extremum in 3x3x3 neighborhood
fn check_extremum(
    val: f32,
    img_prev: &ImageBuffer<Luma<f32>, Vec<f32>>,
    img_curr: &ImageBuffer<Luma<f32>, Vec<f32>>,
    img_next: &ImageBuffer<Luma<f32>, Vec<f32>>,
    x: i32,
    y: i32,
) -> (bool, bool) {
    let mut is_max = true;
    let mut is_min = true;
    
    for dz in -1..=1 {
        let img = match dz {
            -1 => img_prev,
            0 => img_curr,
            1 => img_next,
            _ => unreachable!(),
        };
        
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dz == 0 && dy == 0 && dx == 0 {
                    continue;
                }
                
                let neighbor = get_pixel_clamped(img, x + dx, y + dy);
                
                if val <= neighbor {
                    is_max = false;
                }
                if val >= neighbor {
                    is_min = false;
                }
                
                if !is_max && !is_min {
                    return (false, false);
                }
            }
        }
    }
    
    (is_max, is_min)
}

/// Solve 3x3 linear system using Cramer's rule
fn solve_3x3(a: [[f32; 3]; 3], b: [f32; 3]) -> Option<[f32; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    
    // Use a more lenient threshold for singular matrix detection
    // VLFeat uses a similar approach
    if det.abs() < 1e-7 {
        return None;
    }
    
    let det_x = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);
    
    let det_y = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);
    
    let det_z = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    
    Some([det_x / det, det_y / det, det_z / det])
}

/// Trilinear interpolation for descriptor histogram
fn trilinear_interpolate(hist: &mut [f32; 128], x_bin: f32, y_bin: f32, angle_bin: f32, value: f32) {
    let x_floor = x_bin.floor() as i32;
    let y_floor = y_bin.floor() as i32;
    let angle_floor = angle_bin.floor() as i32;
    
    let dx = x_bin - x_floor as f32;
    let dy = y_bin - y_floor as f32;
    let da = angle_bin - angle_floor as f32;
    
    for i in 0..=1 {
        for j in 0..=1 {
            for k in 0..=1 {
                let xi = (x_floor + i).clamp(0, 3) as usize;
                let yi = (y_floor + j).clamp(0, 3) as usize;
                let ai = ((angle_floor + k).rem_euclid(8)) as usize;
                
                let weight = (if i == 0 { 1.0 - dx } else { dx })
                    * (if j == 0 { 1.0 - dy } else { dy })
                    * (if k == 0 { 1.0 - da } else { da });
                
                let idx = yi * 4 * 8 + xi * 8 + ai;
                if idx < hist.len() {
                    hist[idx] += value * weight;
                }
            }
        }
    }
}
