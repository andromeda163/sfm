//! SIFT Descriptor implementation

/// Descriptor normalization method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// L2 normalization (standard)
    L2,
    /// L1-ROOT normalization (better for SIFT matching)
    /// Default in VLFeat/COLMAP
    L1Root,
}

impl Default for Normalization {
    fn default() -> Self {
        Self::L1Root
    }
}

/// SIFT descriptor (128-dimensional)
pub type Descriptor = [u8; 128];

/// Float descriptor for intermediate computation
pub type DescriptorFloat = [f32; 128];

/// Normalize a descriptor using L2 normalization
pub fn l2_normalize(desc: &mut [f32]) {
    let norm: f32 = desc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for val in desc.iter_mut() {
            *val /= norm;
        }
    }
}

/// Normalize a descriptor using L1-ROOT normalization
/// This is the default in VLFeat/COLMAP and produces better matching results
/// See: "Three things everyone should know to improve object retrieval"
/// by Arandjelovic and Zisserman, CVPR 2012
pub fn l1_root_normalize(desc: &mut [f32]) {
    // L1 normalize
    let l1_norm: f32 = desc.iter().map(|x| x.abs()).sum();
    if l1_norm > 1e-8 {
        for val in desc.iter_mut() {
            *val /= l1_norm;
            // Take square root after L1 normalization
            *val = val.abs().sqrt() * val.signum();
        }
    }
}

/// Normalize descriptor based on the specified normalization type
pub fn normalize(desc: &mut [f32], normalization: Normalization) {
    match normalization {
        Normalization::L2 => l2_normalize(desc),
        Normalization::L1Root => l1_root_normalize(desc),
    }
}

/// Convert float descriptor to unsigned byte [0, 255]
/// Uses the standard SIFT convention: scale from [0, 0.5] to [0, 255]
pub fn to_unsigned_byte(desc: &[f32]) -> Descriptor {
    let mut result = [0u8; 128];
    for (i, &val) in desc.iter().enumerate() {
        if i >= 128 {
            break;
        }
        // Scale from [0, 0.5] to [0, 255] (SIFT convention)
        // Values can be negative after L1_ROOT, so we take absolute value
        let scaled = (val.abs() * 512.0).round() as f32;
        result[i] = scaled.clamp(0.0, 255.0) as u8;
    }
    result
}

/// Transform VLFeat descriptor format to UBC format
/// VLFeat uses a different ordering of the 128 descriptor values
/// This transformation makes descriptors compatible with other implementations
pub fn transform_vlfeat_to_ubc(desc: &Descriptor) -> Descriptor {
    let q = [0, 7, 6, 5, 4, 3, 2, 1];
    let mut result = [0u8; 128];
    
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..8 {
                let src_idx = 8 * (j + 4 * i) + k;
                let dst_idx = 8 * (j + 4 * i) + q[k];
                result[dst_idx] = desc[src_idx];
            }
        }
    }
    
    result
}

/// Compute the L2 distance between two descriptors
pub fn l2_distance(desc1: &Descriptor, desc2: &Descriptor) -> f32 {
    desc1
        .iter()
        .zip(desc2.iter())
        .map(|(a, b)| {
            let diff = (*a as f32) - (*b as f32);
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_root_normalize() {
        let mut desc = [0.0f32; 128];
        desc[0] = 0.5;
        desc[1] = 0.3;
        desc[2] = 0.2;
        l1_root_normalize(&mut desc);
        
        // After L1 norm, values should sum to 1, then sqrt
        assert!((desc[0] - 0.5_f32.sqrt()).abs() < 1e-5);
        assert!((desc[1] - 0.3_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_vlfeat_ubc_transform() {
        let mut input = [0u8; 128];
        for i in 0..128 {
            input[i] = i as u8;
        }
        
        let output = transform_vlfeat_to_ubc(&input);
        
        // The transformation should reorder values
        assert_eq!(output[0], input[0]);  // q[0] = 0
        assert_eq!(output[1], input[7]);  // q[1] = 7
        assert_eq!(output[2], input[6]);  // q[2] = 6
    }
}
