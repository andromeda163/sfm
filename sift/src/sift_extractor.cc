// SIFT Feature Extractor Implementation
// Adapted from COLMAP's feature/sift.cc

#include "sift_extractor.h"

#include <algorithm>
#include <array>
#include <cstring>

// VLFeat SIFT
extern "C" {
#include "vlfeat/sift.h"
}

namespace sift {

SiftExtractor::SiftExtractor(const SiftOptions& options)
    : options_(options), sift_filter_(nullptr) {
  // Options validation
  if (!options_.Check()) {
    throw std::invalid_argument("Invalid SIFT options");
  }
}

SiftExtractor::~SiftExtractor() {
  if (sift_filter_) {
    vl_sift_delete(static_cast<VlSiftFilt*>(sift_filter_));
  }
}

bool SiftExtractor::Extract(const float* image_data,
                            int width,
                            int height,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  if (!image_data || width <= 0 || height <= 0) {
    return false;
  }

  // Create or resize SIFT filter if needed
  VlSiftFilt* sift = static_cast<VlSiftFilt*>(sift_filter_);
  if (!sift || sift->width != width || sift->height != height) {
    if (sift) {
      vl_sift_delete(sift);
    }
    sift = vl_sift_new(width, height,
                       options_.num_octaves,
                       options_.octave_resolution,
                       options_.first_octave);
    if (!sift) {
      return false;
    }
    sift_filter_ = sift;
  }

  // Set thresholds
  vl_sift_set_peak_thresh(sift, options_.peak_threshold);
  vl_sift_set_edge_thresh(sift, options_.edge_threshold);

  // Process image through octaves
  std::vector<size_t> level_num_features;
  std::vector<FeatureKeypoints> level_keypoints;
  std::vector<FeatureDescriptorsFloat> level_descriptors;

  bool first_octave = true;
  while (true) {
    if (first_octave) {
      if (vl_sift_process_first_octave(sift, image_data)) {
        break;
      }
      first_octave = false;
    } else {
      if (vl_sift_process_next_octave(sift)) {
        break;
      }
    }

    // Detect keypoints
    vl_sift_detect(sift);

    const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift);
    const int num_keypoints = vl_sift_get_nkeypoints(sift);
    if (num_keypoints == 0) {
      continue;
    }

    // Extract features with different orientations per DOG level
    size_t level_idx = 0;
    int prev_level = -1;
    FeatureDescriptorsFloat desc(1, 128);

    for (int i = 0; i < num_keypoints; ++i) {
      if (vl_keypoints[i].is != prev_level) {
        if (i > 0) {
          // Resize containers of previous DOG level
          level_keypoints.back().resize(level_idx);
          if (descriptors) {
            level_descriptors.back().conservativeResize(level_idx, 128);
          }
        }

        // Add containers for new DOG level
        level_idx = 0;
        level_num_features.push_back(0);
        level_keypoints.emplace_back(options_.max_num_orientations * num_keypoints);
        if (descriptors) {
          level_descriptors.emplace_back(options_.max_num_orientations * num_keypoints, 128);
        }
      }

      level_num_features.back() += 1;
      prev_level = vl_keypoints[i].is;

      // Extract feature orientations
      double angles[4];
      int num_orientations;
      if (options_.upright) {
        num_orientations = 1;
        angles[0] = 0.0;
      } else {
        num_orientations = vl_sift_calc_keypoint_orientations(sift, angles, &vl_keypoints[i]);
      }

      // Limit number of orientations
      const int num_used_orientations =
          std::min(num_orientations, options_.max_num_orientations);

      for (int o = 0; o < num_used_orientations; ++o) {
        level_keypoints.back()[level_idx] = FeatureKeypoint(
            vl_keypoints[i].x + 0.5f,
            vl_keypoints[i].y + 0.5f,
            vl_keypoints[i].sigma,
            static_cast<float>(angles[o]));

        if (descriptors) {
          vl_sift_calc_keypoint_descriptor(sift, desc.data(), &vl_keypoints[i], angles[o]);

          // Normalize descriptor
          if (options_.normalization == SiftOptions::Normalization::L2) {
            L2NormalizeFeatureDescriptors(&desc);
          } else {
            L1RootNormalizeFeatureDescriptors(&desc);
          }

          level_descriptors.back().row(level_idx) = desc;
        }

        level_idx += 1;
      }
    }

    // Resize containers for last DOG level in octave
    level_keypoints.back().resize(level_idx);
    if (descriptors) {
      level_descriptors.back().conservativeResize(level_idx, 128);
    }
  }

  // Determine how many DOG levels to keep to satisfy max_num_features option
  int first_level_to_keep = 0;
  int num_features = 0;
  int num_features_with_orientations = 0;
  for (int i = static_cast<int>(level_keypoints.size()) - 1; i >= 0; --i) {
    num_features += level_num_features[i];
    num_features_with_orientations += static_cast<int>(level_keypoints[i].size());
    if (num_features > options_.max_num_features) {
      first_level_to_keep = i;
      break;
    }
  }

  // Extract the features to be kept
  keypoints->clear();
  keypoints->reserve(num_features_with_orientations);
  for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
    for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
      keypoints->push_back(level_keypoints[i][j]);
    }
  }

  // Compute the descriptors for the detected keypoints
  if (descriptors) {
    descriptors->resize(keypoints->size(), 128);
    size_t k = 0;
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        // Convert to unsigned byte
        for (int c = 0; c < 128; ++c) {
          float val = std::round(512.0f * level_descriptors[i](j, c));
          (*descriptors)(k, c) = static_cast<uint8_t>(
              std::max(0.0f, std::min(255.0f, val)));
        }
        k++;
      }
    }

    // Transform from VLFeat to UBC format
    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
  }

  return true;
}

FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(), 128);
  const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};

  for (Eigen::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
              vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}

}  // namespace sift
