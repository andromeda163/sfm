//! SIFT Feature Extractor
//!
//! A Rust implementation of the SIFT (Scale-Invariant Feature Transform) algorithm
//! matching the behavior of COLMAP/VLFeat for reproducible results.

pub mod keypoint;
pub mod descriptor;
pub mod pyramid;
pub mod extractor;

pub use extractor::{SiftExtractor, SiftOptions};
pub use keypoint::Keypoint;
pub use descriptor::{Descriptor, Normalization};

/// Load an image and convert to grayscale f32 in [0, 1]
pub fn load_image_grayscale(path: &str) -> Option<(Vec<f32>, u32, u32)> {
    use image::GenericImageView;
    
    let img = image::open(path).ok()?;
    let (width, height) = (img.width(), img.height());
    
    let gray: Vec<f32> = img
        .to_luma8()
        .pixels()
        .map(|p| p[0] as f32 / 255.0)
        .collect();
    
    Some((gray, width, height))
}
