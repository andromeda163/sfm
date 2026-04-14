cmake_minimum_required(VERSION 3.12)
project(StandaloneSIFT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(SIMD_ENABLED "Enable SIMD optimizations" ON)

# Find dependencies
find_package(Eigen3 REQUIRED)

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -O3)
  if(SIMD_ENABLED)
    # Check for x86 architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|amd64")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse2")
      add_compile_definitions(VL_DISABLE_AVX)  # Disable AVX for compatibility
    endif()
  else()
    add_compile_definitions(VL_DISABLE_AVX VL_DISABLE_SSE2)
  endif()
endif()

# VLFeat library (SIFT implementation)
add_library(vlfeat_sift STATIC
  thirdparty/vlfeat/sift.c
  thirdparty/vlfeat/generic.c
  thirdparty/vlfeat/mathop.c
  thirdparty/vlfeat/random.c
  thirdparty/vlfeat/host.c
  thirdparty/vlfeat/imopv.c
  thirdparty/vlfeat/scalespace.c
)

target_include_directories(vlfeat_sift PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

# SSE2/AVX sources (conditional)
if(SIMD_ENABLED AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|amd64")
  # SSE2 sources
  target_sources(vlfeat_sift PRIVATE
    thirdparty/vlfeat/mathop_sse2.c
    thirdparty/vlfeat/imopv_sse2.c
  )
  set_source_files_properties(
    thirdparty/vlfeat/mathop_sse2.c
    thirdparty/vlfeat/imopv_sse2.c
    PROPERTIES COMPILE_FLAGS "-msse2"
  )
  
  # AVX sources (optional, can be disabled for compatibility)
  option(AVX_ENABLED "Enable AVX optimizations" OFF)
  if(AVX_ENABLED)
    target_sources(vlfeat_sift PRIVATE
      thirdparty/vlfeat/mathop_avx.c
    )
    set_source_files_properties(
      thirdparty/vlfeat/mathop_avx.c
      PROPERTIES COMPILE_FLAGS "-mavx"
    )
  else()
    target_compile_definitions(vlfeat_sift PRIVATE VL_DISABLE_AVX)
  endif()
endif()

# OpenMP support (optional)
find_package(OpenMP)
if(OpenMP_C_FOUND)
  target_link_libraries(vlfeat_sift PRIVATE OpenMP::OpenMP_C)
else()
  target_compile_definitions(vlfeat_sift PRIVATE VL_DISABLE_OPENMP)
endif()

# Main executable
add_executable(sift_extract
  src/main.cc
  src/sift_extractor.cc
)

target_include_directories(sift_extract PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

target_link_libraries(sift_extract PRIVATE
  vlfeat_sift
  Eigen3::Eigen
)

# Installation
install(TARGETS sift_extract DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "Standalone SIFT Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD enabled: ${SIMD_ENABLED}")
message(STATUS "  OpenMP found: ${OpenMP_C_FOUND}")
message(STATUS "")
