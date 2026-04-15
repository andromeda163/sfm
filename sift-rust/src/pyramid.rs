//! Gaussian and Difference-of-Gaussians pyramid construction

use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use image::imageops::FilterType;
use imageproc::filter::gaussian_blur_f32;
use rayon::prelude::*;

/// Represents a single octave in the Gaussian pyramid
pub type GaussianOctave = Vec<GrayImage>;

/// Represents a single octave in the DoG pyramid (f32 images)
pub type DogOctave = Vec<ImageBuffer<Luma<f32>, Vec<f32>>>;

/// Gaussian pyramid (vector of octaves)
pub type GaussianPyramid = Vec<GaussianOctave>;

/// DoG pyramid (vector of octaves)
pub type DogPyramid = Vec<DogOctave>;

/// Build the Gaussian pyramid from a base image
pub fn build_gaussian_pyramid(
    base_image: &GrayImage,
    num_octaves: usize,
    num_intervals: u32,
    sigma: f32,
) -> GaussianPyramid {
    let mut pyramid = Vec::with_capacity(num_octaves);
    let mut current_octave_base = base_image.clone();
    
    // Precompute target sigmas for each level in an octave
    let k = 2.0_f32.powf(1.0 / num_intervals as f32);
    let octave_target_sigmas: Vec<f32> = (0..(num_intervals + 3))
        .map(|s| sigma * k.powi(s as i32))
        .collect();
    
    for octave_idx in 0..num_octaves {
        let mut octave_images = Vec::with_capacity((num_intervals + 3) as usize);
        octave_images.push(current_octave_base.clone());
        
        // Absolute sigma at the start of this octave
        let mut prev_sigma_abs = sigma * 2.0_f32.powi(octave_idx as i32);
        
        // Build each level of the octave
        for s_idx in 1..(num_intervals + 3) as usize {
            let target_sigma_abs = octave_target_sigmas[s_idx] * 2.0_f32.powi(octave_idx as i32);
            
            // Blur needed to go from prev_sigma to target_sigma
            let blur_sigma = (target_sigma_abs.powi(2) - prev_sigma_abs.powi(2)).sqrt();
            
            let blurred = if blur_sigma < 1e-4 {
                octave_images.last().unwrap().clone()
            } else {
                gaussian_blur_f32(octave_images.last().unwrap(), blur_sigma)
            };
            
            octave_images.push(blurred);
            prev_sigma_abs = target_sigma_abs;
        }
        
        pyramid.push(octave_images);
        
        // Prepare base for next octave (downsample by 2x)
        if octave_idx < num_octaves - 1 {
            let next_base_idx = num_intervals as usize;
            let next_base = &pyramid.last().unwrap()[next_base_idx];
            current_octave_base = resize_image(next_base, next_base.width() / 2, next_base.height() / 2);
        }
    }
    
    pyramid
}

/// Build the Difference-of-Gaussians pyramid from the Gaussian pyramid
pub fn build_dog_pyramid(gaussian_pyramid: &GaussianPyramid) -> DogPyramid {
    gaussian_pyramid
        .par_iter()
        .map(|octave| {
            // Convert to f32 and compute differences
            let octave_f32: Vec<ImageBuffer<Luma<f32>, Vec<f32>>> = octave
                .iter()
                .map(|img| convert_u8_to_f32(img))
                .collect();
            
            (0..(octave_f32.len() - 1))
                .map(|i| subtract_images(&octave_f32[i + 1], &octave_f32[i]))
                .collect()
        })
        .collect()
}

/// Upsample an image by 2x using bilinear interpolation
pub fn upsample_image(img: &GrayImage) -> GrayImage {
    let new_width = img.width() * 2;
    let new_height = img.height() * 2;
    resize_image(img, new_width, new_height)
}

/// Resize an image to the specified dimensions
pub fn resize_image(img: &GrayImage, new_width: u32, new_height: u32) -> GrayImage {
    let dyn_img = DynamicImage::ImageLuma8(img.clone());
    dyn_img.resize_exact(new_width, new_height, FilterType::Lanczos3).into_luma8()
}

/// Convert u8 grayscale image to f32 in [0, 1]
pub fn convert_u8_to_f32(img: &GrayImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = img.dimensions();
    let pixels: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0).collect();
    ImageBuffer::from_raw(width, height, pixels).unwrap()
}

/// Subtract two f32 images
pub fn subtract_images(
    img1: &ImageBuffer<Luma<f32>, Vec<f32>>,
    img2: &ImageBuffer<Luma<f32>, Vec<f32>>,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = img1.dimensions();
    let pixels: Vec<f32> = img1
        .as_raw()
        .iter()
        .zip(img2.as_raw().iter())
        .map(|(p1, p2)| p1 - p2)
        .collect();
    ImageBuffer::from_raw(width, height, pixels).unwrap()
}

/// Get pixel value with boundary clamping
#[inline(always)]
pub fn get_pixel_clamped(img: &ImageBuffer<Luma<f32>, Vec<f32>>, x: i32, y: i32) -> f32 {
    let x_clamped = x.clamp(0, img.width() as i32 - 1) as u32;
    let y_clamped = y.clamp(0, img.height() as i32 - 1) as u32;
    img.get_pixel(x_clamped, y_clamped)[0]
}

/// Get pixel value from u8 image with boundary clamping, normalized to [0, 1]
#[inline(always)]
pub fn get_pixel_u8_clamped(img: &GrayImage, x: i32, y: i32) -> f32 {
    let x_clamped = x.clamp(0, img.width() as i32 - 1) as u32;
    let y_clamped = y.clamp(0, img.height() as i32 - 1) as u32;
    img.get_pixel(x_clamped, y_clamped)[0] as f32 / 255.0
}

/// Bilinear interpolation for pixel value
#[inline(always)]
pub fn get_pixel_bilinear(img: &GrayImage, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    
    let q00 = get_pixel_u8_clamped(img, x0, y0);
    let q10 = get_pixel_u8_clamped(img, x1, y0);
    let q01 = get_pixel_u8_clamped(img, x0, y1);
    let q11 = get_pixel_u8_clamped(img, x1, y1);
    
    q00 * (1.0 - dx) * (1.0 - dy) + q10 * dx * (1.0 - dy) + q01 * (1.0 - dx) * dy + q11 * dx * dy
}
