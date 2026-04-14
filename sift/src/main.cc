// Standalone SIFT Feature Extraction Tool
// Extracts SIFT features from images and outputs them to files

#include "feature_types.h"
#include "sift_extractor.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace {

// Load image and convert to grayscale using Rec. 709 (same as COLMAP)
// This ensures identical results to COLMAP's Bitmap::CloneAsGrey()
std::vector<float> LoadGrayscaleImage(const std::string& path, int& width, int& height) {
  int channels;
  // Always load as RGB to ensure consistent grayscale conversion
  unsigned char* rgb = stbi_load(path.c_str(), &width, &height, &channels, 3);
  if (!rgb) {
    return {};
  }

  std::vector<float> result(width * height);
  // Convert to grayscale using Rec. 709 coefficients (same as COLMAP)
  // Y = 0.2126*R + 0.7152*G + 0.0722*B
  for (int i = 0; i < width * height; ++i) {
    float gray = std::round(0.2126f * rgb[3*i] + 0.7152f * rgb[3*i+1] + 0.0722f * rgb[3*i+2]);
    result[i] = gray / 255.0f;
  }

  stbi_image_free(rgb);
  return result;
}

// Load raw grayscale from COLMAP's dumped binary file
std::vector<float> LoadRawGrayscale(const std::string& path, int& width, int& height) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open raw grayscale file: " << path << std::endl;
    return {};
  }
  
  file.read(reinterpret_cast<char*>(&width), sizeof(int));
  file.read(reinterpret_cast<char*>(&height), sizeof(int));
  
  std::vector<uint8_t> raw_data(width * height);
  file.read(reinterpret_cast<char*>(raw_data.data()), width * height);
  file.close();
  
  std::vector<float> result(width * height);
  for (int i = 0; i < width * height; ++i) {
    result[i] = static_cast<float>(raw_data[i]) / 255.0f;
  }
  
  return result;
}

// Write keypoints to COLMAP text format
void WriteKeypointsText(const std::string& path,
                        const sift::FeatureKeypoints& keypoints,
                        const sift::FeatureDescriptors& descriptors) {
  std::ofstream file(path);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << path << std::endl;
    return;
  }

  // Header: num_features descriptor_dim
  file << keypoints.size() << " " << 128 << "\n";

  // Each feature: x y scale orientation d1 d2 ... d128
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const auto& kp = keypoints[i];
    file << kp.x << " " << kp.y << " " << kp.scale << " " << kp.orientation;
    for (int j = 0; j < 128; ++j) {
      file << " " << static_cast<int>(descriptors(i, j));
    }
    file << "\n";
  }
}

// Write keypoints to CSV format
void WriteKeypointsCSV(const std::string& path,
                       const sift::FeatureKeypoints& keypoints) {
  std::ofstream file(path);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << path << std::endl;
    return;
  }

  file << "x,y,scale,orientation\n";
  for (const auto& kp : keypoints) {
    file << kp.x << "," << kp.y << "," << kp.scale << "," << kp.orientation << "\n";
  }
}

// Write descriptors to binary file
void WriteDescriptorsBinary(const std::string& path,
                            const sift::FeatureDescriptors& descriptors) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << path << std::endl;
    return;
  }

  // Write dimensions
  int rows = static_cast<int>(descriptors.rows());
  int cols = static_cast<int>(descriptors.cols());
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

  // Write data
  file.write(reinterpret_cast<const char*>(descriptors.data()),
             descriptors.size());
}

void PrintUsage(const char* program_name) {
  std::cout << "Standalone SIFT Feature Extractor\n"
            << "\n"
            << "Usage: " << program_name << " [options] <image_path> <output_dir>\n"
            << "\n"
            << "Options:\n"
            << "  --max_features N    Maximum number of features (default: 8192)\n"
            << "  --first_octave N    First octave (default: -1, -1 upsamples)\n"
            << "  --num_octaves N     Number of octaves (default: 4)\n"
            << "  --octave_levels N   Levels per octave (default: 3)\n"
            << "  --peak_thresh F     Peak threshold (default: 0.00667)\n"
            << "  --edge_thresh F     Edge threshold (default: 10.0)\n"
            << "  --max_orientations N Max orientations per keypoint (default: 2)\n"
            << "  --normalization N   Normalization: L1_ROOT or L2 (default: L1_ROOT)\n"
            << "  --upright           Fix orientation to 0 (upright features)\n"
            << "  --binary            Also output binary descriptor file\n"
            << "  --raw_grayscale     Input is raw grayscale .bin file from COLMAP\n"
            << "  --help              Show this help message\n"
            << "\n"
            << "Output files:\n"
            << "  <output_dir>/<image_name>_keypoints.txt  - Keypoints in COLMAP format\n"
            << "  <output_dir>/<image_name>_keypoints.csv  - Keypoints in CSV format\n"
            << "  <output_dir>/<image_name>_descriptors.bin - Binary descriptors (with --binary)\n"
            << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  // Default options
  sift::SiftOptions options;
  bool output_binary = false;
  bool raw_grayscale = false;
  std::string image_path;
  std::string output_dir;

  // Parse arguments
  int arg_idx = 1;
  while (arg_idx < argc) {
    std::string arg = argv[arg_idx];

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else if (arg == "--max_features" && arg_idx + 1 < argc) {
      options.max_num_features = std::stoi(argv[++arg_idx]);
    } else if (arg == "--first_octave" && arg_idx + 1 < argc) {
      options.first_octave = std::stoi(argv[++arg_idx]);
    } else if (arg == "--num_octaves" && arg_idx + 1 < argc) {
      options.num_octaves = std::stoi(argv[++arg_idx]);
    } else if (arg == "--octave_levels" && arg_idx + 1 < argc) {
      options.octave_resolution = std::stoi(argv[++arg_idx]);
      options.peak_threshold = 0.02 / options.octave_resolution;
    } else if (arg == "--peak_thresh" && arg_idx + 1 < argc) {
      options.peak_threshold = std::stod(argv[++arg_idx]);
    } else if (arg == "--edge_thresh" && arg_idx + 1 < argc) {
      options.edge_threshold = std::stod(argv[++arg_idx]);
    } else if (arg == "--max_orientations" && arg_idx + 1 < argc) {
      options.max_num_orientations = std::stoi(argv[++arg_idx]);
    } else if (arg == "--normalization" && arg_idx + 1 < argc) {
      std::string norm = argv[++arg_idx];
      if (norm == "L2") {
        options.normalization = sift::SiftOptions::Normalization::L2;
      } else {
        options.normalization = sift::SiftOptions::Normalization::L1_ROOT;
      }
    } else if (arg == "--upright") {
      options.upright = true;
    } else if (arg == "--binary") {
      output_binary = true;
    } else if (arg == "--raw_grayscale") {
      raw_grayscale = true;
    } else if (arg[0] != '-') {
      // Positional argument
      if (image_path.empty()) {
        image_path = arg;
      } else if (output_dir.empty()) {
        output_dir = arg;
      }
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      return 1;
    }
    ++arg_idx;
  }

  // Validate arguments
  if (image_path.empty() || output_dir.empty()) {
    std::cerr << "Error: Missing required arguments\n" << std::endl;
    PrintUsage(argv[0]);
    return 1;
  }

  // Validate options
  if (!options.Check()) {
    std::cerr << "Error: Invalid SIFT options" << std::endl;
    return 1;
  }

  // Create output directory
  std::filesystem::create_directories(output_dir);

  // Load image
  std::cout << "Loading image: " << image_path << std::endl;
  int width, height;
  std::vector<float> image_data;
  
  if (raw_grayscale) {
    image_data = LoadRawGrayscale(image_path, width, height);
  } else {
    image_data = LoadGrayscaleImage(image_path, width, height);
  }
  
  if (image_data.empty()) {
    std::cerr << "Error: Could not load image: " << image_path << std::endl;
    return 1;
  }
  std::cout << "Image size: " << width << " x " << height << std::endl;

  // Create extractor
  std::cout << "Extracting SIFT features (max " << options.max_num_features << ")..." << std::endl;
  sift::SiftExtractor extractor(options);

  // Extract features
  sift::FeatureKeypoints keypoints;
  sift::FeatureDescriptors descriptors;
  if (!extractor.Extract(image_data.data(), width, height, &keypoints, &descriptors)) {
    std::cerr << "Error: Feature extraction failed" << std::endl;
    return 1;
  }

  std::cout << "Extracted " << keypoints.size() << " features" << std::endl;

  // Generate output paths
  std::filesystem::path input_path(image_path);
  std::string stem = input_path.stem().string();
  std::filesystem::path output_txt = std::filesystem::path(output_dir) / (stem + "_keypoints.txt");
  std::filesystem::path output_csv = std::filesystem::path(output_dir) / (stem + "_keypoints.csv");
  std::filesystem::path output_bin = std::filesystem::path(output_dir) / (stem + "_descriptors.bin");

  // Write output files
  WriteKeypointsText(output_txt.string(), keypoints, descriptors);
  WriteKeypointsCSV(output_csv.string(), keypoints);
  if (output_binary) {
    WriteDescriptorsBinary(output_bin.string(), descriptors);
  }

  std::cout << "\n=== Extraction Summary ===" << std::endl;
  std::cout << "Image: " << image_path << std::endl;
  std::cout << "Size: " << width << " x " << height << std::endl;
  std::cout << "Features: " << keypoints.size() << std::endl;
  std::cout << "Descriptor dim: 128" << std::endl;
  std::cout << "Output text: " << output_txt << std::endl;
  std::cout << "Output CSV: " << output_csv << std::endl;
  if (output_binary) {
    std::cout << "Output binary: " << output_bin << std::endl;
  }

  return 0;
}
