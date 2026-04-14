// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Standalone SIFT feature extraction tool.
// Extracts SIFT features from images and outputs them to text files.

#include "colmap/feature/sift.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace colmap;

void WriteSiftFeaturesToTextFile(const std::filesystem::path& path,
                                  const FeatureKeypoints& keypoints,
                                  const FeatureDescriptors& descriptors) {
  std::ofstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Could not open file for writing: " << path;
    return;
  }

  // Write header: num_features descriptor_dim
  file << keypoints.size() << " " << descriptors.data.cols() << "\n";

  // Write each feature: x y scale orientation d1 d2 ... d128
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const auto& kp = keypoints[i];
    file << kp.x << " " << kp.y << " " << kp.ComputeScale() << " "
         << kp.ComputeOrientation();
    for (Eigen::Index j = 0; j < descriptors.data.cols(); ++j) {
      file << " " << static_cast<int>(descriptors.data(i, j));
    }
    file << "\n";
  }
  file.close();
}

void WriteKeypointsToCSV(const std::filesystem::path& path,
                         const FeatureKeypoints& keypoints) {
  std::ofstream file(path);
  file << "x,y,scale,orientation\n";
  for (const auto& kp : keypoints) {
    file << kp.x << "," << kp.y << "," << kp.ComputeScale() << ","
         << kp.ComputeOrientation() << "\n";
  }
  file.close();
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " [options] image_path output_dir\n"
              << "\n"
              << "Options:\n"
              << "  --max_features N    Maximum number of features (default: 8192)\n"
              << "  --first_octave N    First octave (default: -1)\n"
              << "  --num_octaves N     Number of octaves (default: 4)\n"
              << "  --peak_thresh F     Peak threshold (default: 0.02/3)\n"
              << "  --edge_thresh F     Edge threshold (default: 10.0)\n"
              << "  --normalization N   Normalization: L1_ROOT or L2 (default: L1_ROOT)\n"
              << "  --upright           Fix orientation to 0 (upright features)\n"
              << "\n"
              << "Output files:\n"
              << "  <output_dir>/<image_name>_keypoints.txt  - Keypoints in COLMAP format\n"
              << "  <output_dir>/<image_name>_keypoints.csv  - Keypoints in CSV format\n"
              << std::endl;
    return EXIT_FAILURE;
  }

  // Parse arguments
  std::string image_path;
  std::string output_dir;
  int max_features = 8192;
  int first_octave = -1;
  int num_octaves = 4;
  double peak_thresh = 0.02 / 3;  // Default from SiftExtractionOptions
  double edge_thresh = 10.0;
  std::string normalization = "L1_ROOT";
  bool upright = false;

  // Parse options
  int arg_idx = 1;
  while (arg_idx < argc - 2) {
    std::string arg = argv[arg_idx];
    if (arg == "--max_features" && arg_idx + 1 < argc - 2) {
      max_features = std::stoi(argv[++arg_idx]);
    } else if (arg == "--first_octave" && arg_idx + 1 < argc - 2) {
      first_octave = std::stoi(argv[++arg_idx]);
    } else if (arg == "--num_octaves" && arg_idx + 1 < argc - 2) {
      num_octaves = std::stoi(argv[++arg_idx]);
    } else if (arg == "--peak_thresh" && arg_idx + 1 < argc - 2) {
      peak_thresh = std::stod(argv[++arg_idx]);
    } else if (arg == "--edge_thresh" && arg_idx + 1 < argc - 2) {
      edge_thresh = std::stod(argv[++arg_idx]);
    } else if (arg == "--normalization" && arg_idx + 1 < argc - 2) {
      normalization = argv[++arg_idx];
    } else if (arg == "--upright") {
      upright = true;
    } else {
      LOG(ERROR) << "Unknown option: " << arg;
      return EXIT_FAILURE;
    }
    ++arg_idx;
  }

  // Last two arguments are image_path and output_dir
  image_path = argv[argc - 2];
  output_dir = argv[argc - 1];

  // Create output directory if it doesn't exist
  std::filesystem::create_directories(output_dir);

  // Configure SIFT extraction options
  FeatureExtractionOptions options;
  options.type = FeatureExtractorType::SIFT;
  options.use_gpu = false;  // CPU-only for portability
  options.sift->max_num_features = max_features;
  options.sift->first_octave = first_octave;
  options.sift->num_octaves = num_octaves;
  options.sift->peak_threshold = peak_thresh;
  options.sift->edge_threshold = edge_thresh;
  options.sift->upright = upright;
  
  if (normalization == "L2") {
    options.sift->normalization = SiftExtractionOptions::Normalization::L2;
  } else {
    options.sift->normalization = SiftExtractionOptions::Normalization::L1_ROOT;
  }

  // Validate options
  if (!options.Check()) {
    LOG(ERROR) << "Invalid options";
    return EXIT_FAILURE;
  }

  // Load image
  Bitmap bitmap;
  if (!bitmap.Read(image_path, false)) {  // false = grayscale
    LOG(ERROR) << "Could not read image: " << image_path;
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Image size: " << bitmap.Width() << " x " << bitmap.Height();

  // Dump grayscale for comparison with standalone tool
  {
    std::filesystem::path input_path(image_path);
    std::string stem = input_path.stem().string();
    std::filesystem::path gray_path = std::filesystem::path(output_dir) / (stem + "_gray.bin");
    
    std::ofstream gray_file(gray_path, std::ios::binary);
    int w = bitmap.Width();
    int h = bitmap.Height();
    gray_file.write(reinterpret_cast<const char*>(&w), sizeof(int));
    gray_file.write(reinterpret_cast<const char*>(&h), sizeof(int));
    const auto& data = bitmap.RowMajorData();
    gray_file.write(reinterpret_cast<const char*>(data.data()), w * h);
    gray_file.close();
    
    LOG(INFO) << "Dumped grayscale to: " << gray_path;
  }
  LOG(INFO) << "Extracting SIFT features with max " << max_features << " features...";

  // Create SIFT extractor
  auto extractor = CreateSiftFeatureExtractor(options);
  if (!extractor) {
    LOG(ERROR) << "Could not create SIFT feature extractor";
    return EXIT_FAILURE;
  }

  // Extract features
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  if (!extractor->Extract(bitmap, &keypoints, &descriptors)) {
    LOG(ERROR) << "Feature extraction failed";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Extracted " << keypoints.size() << " features";

  // Generate output filenames
  std::filesystem::path input_path(image_path);
  std::string stem = input_path.stem().string();
  std::filesystem::path output_txt = std::filesystem::path(output_dir) / (stem + "_keypoints.txt");
  std::filesystem::path output_csv = std::filesystem::path(output_dir) / (stem + "_keypoints.csv");

  // Write output files
  WriteSiftFeaturesToTextFile(output_txt, keypoints, descriptors);
  WriteKeypointsToCSV(output_csv, keypoints);

  LOG(INFO) << "Wrote keypoints to: " << output_txt;
  LOG(INFO) << "Wrote CSV to: " << output_csv;

  // Print summary statistics
  std::cout << "\n=== Extraction Summary ===\n";
  std::cout << "Image: " << image_path << "\n";
  std::cout << "Size: " << bitmap.Width() << " x " << bitmap.Height() << "\n";
  std::cout << "Features extracted: " << keypoints.size() << "\n";
  std::cout << "Descriptor dimension: " << descriptors.data.cols() << "\n";
  std::cout << "Output text file: " << output_txt << "\n";
  std::cout << "Output CSV file: " << output_csv << "\n";

  return EXIT_SUCCESS;
}
