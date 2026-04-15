//! Keypoint data structures

use std::fmt;

/// A SIFT keypoint with position, scale, and orientation
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    /// X coordinate (in original image coordinates)
    pub x: f32,
    /// Y coordinate (in original image coordinates)
    pub y: f32,
    /// Scale (sigma in original image coordinates)
    pub scale: f32,
    /// Orientation in radians [-PI, PI]
    pub orientation: f32,
    /// Octave index where the keypoint was found
    pub octave: i32,
    /// Layer index within the octave
    pub layer: i32,
    /// DoG response value (for sorting)
    pub response: f32,
}

impl Keypoint {
    pub fn new(x: f32, y: f32, scale: f32, orientation: f32) -> Self {
        Self {
            x,
            y,
            scale,
            orientation,
            octave: 0,
            layer: 0,
            response: 0.0,
        }
    }
    
    /// Create a keypoint with octave/layer info
    pub fn with_octave_layer(mut self, octave: i32, layer: i32) -> Self {
        self.octave = octave;
        self.layer = layer;
        self
    }
    
    /// Create a keypoint with response value
    pub fn with_response(mut self, response: f32) -> Self {
        self.response = response;
        self
    }
}

impl fmt::Display for Keypoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Keypoint(x={:.2}, y={:.2}, scale={:.2}, orientation={:.2})",
            self.x, self.y, self.scale, self.orientation
        )
    }
}

/// A collection of keypoints
pub type Keypoints = Vec<Keypoint>;
