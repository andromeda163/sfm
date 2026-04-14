// Minimal feature types for SIFT extraction
// Adapted from COLMAP's feature/types.h

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace sift {

// Feature keypoint structure
struct FeatureKeypoint {
  float x;          // x position (origin at upper-left, 0.5, 0.5 is pixel center)
  float y;          // y position
  float scale;      // scale (sigma)
  float orientation; // orientation in radians [-pi, pi]

  FeatureKeypoint() : x(0), y(0), scale(0), orientation(0) {}

  FeatureKeypoint(float x, float y, float scale, float orientation)
      : x(x), y(y), scale(scale), orientation(orientation) {}

  // Compute scale from affine shape (for compatibility)
  float ComputeScale() const { return scale; }
  
  // Compute orientation (already stored)
  float ComputeOrientation() const { return orientation; }
};

using FeatureKeypoints = std::vector<FeatureKeypoint>;

// Feature descriptors - 128-dimensional SIFT descriptor
// Stored as uint8_t values in range [0, 255]
using FeatureDescriptors = Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor>;

// Float descriptors for intermediate computation
using FeatureDescriptorsFloat = Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>;

// L2 normalize feature descriptors
inline void L2NormalizeFeatureDescriptors(FeatureDescriptorsFloat* descriptors) {
  descriptors->rowwise().normalize();
}

// L1-root normalize feature descriptors (better for SIFT)
// See "Three things everyone should know to improve object retrieval",
// Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
inline void L1RootNormalizeFeatureDescriptors(FeatureDescriptorsFloat* descriptors) {
  for (Eigen::Index r = 0; r < descriptors->rows(); ++r) {
    descriptors->row(r) *= 1.0f / descriptors->row(r).lpNorm<1>();
    descriptors->row(r) = descriptors->row(r).array().sqrt();
  }
}

// Convert float descriptors to unsigned byte [0, 255]
inline FeatureDescriptors FeatureDescriptorsToUnsignedByte(
    const FeatureDescriptorsFloat& descriptors) {
  FeatureDescriptors result(descriptors.rows(), 128);
  for (Eigen::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::Index c = 0; c < 128; ++c) {
      // Scale from [0, 0.5] to [0, 255] (SIFT convention)
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      result(r, c) = static_cast<uint8_t>(
          std::max(0.0f, std::min(255.0f, scaled_value)));
    }
  }
  return result;
}

}  // namespace sift
