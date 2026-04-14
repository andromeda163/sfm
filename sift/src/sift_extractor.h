// SIFT Feature Extractor
// Adapted from COLMAP's feature/sift.cc - CPU implementation using VLFeat

#pragma once

#include "feature_types.h"

#include <memory>
#include <string>

namespace sift {

// SIFT extraction options
struct SiftOptions {
  // Maximum number of features to detect, keeping larger-scale features
  int max_num_features = 8192;

  // First octave in the pyramid, i.e. -1 upsamples the image by one level
  int first_octave = -1;

  // Number of octaves (default: 4)
  int num_octaves = 4;

  // Number of levels per octave (default: 3)
  int octave_resolution = 3;

  // Peak threshold for detection (default: 0.02 / octave_resolution)
  double peak_threshold = 0.02 / 3.0;

  // Edge threshold for detection (default: 10.0)
  double edge_threshold = 10.0;

  // Maximum number of orientations per keypoint (default: 2)
  int max_num_orientations = 2;

  // Fix the orientation to 0 for upright features
  bool upright = false;

  // Descriptor normalization type
  enum class Normalization {
    L1_ROOT,  // L1 normalize then sqrt (default, better)
    L2        // Standard L2 normalization
  };
  Normalization normalization = Normalization::L1_ROOT;

  // Validate options
  bool Check() const {
    if (max_num_features <= 0) return false;
    if (octave_resolution <= 0) return false;
    if (peak_threshold <= 0.0) return false;
    if (edge_threshold <= 0.0) return false;
    if (max_num_orientations <= 0) return false;
    return true;
  }
};

// SIFT feature extractor using VLFeat
class SiftExtractor {
 public:
  explicit SiftExtractor(const SiftOptions& options);
  ~SiftExtractor();

  // Extract SIFT features from a grayscale image
  // image_data: grayscale image as float array, values in [0, 1]
  // width, height: image dimensions
  // Returns true on success
  bool Extract(const float* image_data,
               int width,
               int height,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors);

 private:
  // VLFeat SIFT filter handle
  void* sift_filter_;  // VlSiftFilt*
  SiftOptions options_;
};

// VLFeat uses a different convention for descriptor ordering
// This transforms VLFeat format to the standard UBC format
FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors);

}  // namespace sift
