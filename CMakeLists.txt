cmake_minimum_required(VERSION 2.8.12)
project(deep_3d_descriptor)

if(NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

SET(PCL_DIR "" CACHE STRING "path to PCLConfig.cmake for 1.8")
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/python_cpp/
)

add_library(deep_3d_descriptor SHARED
  src/deep_3d_descriptor.cpp
  python_cpp/gen-cpp/get_descriptors.cpp
  python_cpp/gen-cpp/python_cpp_types.cpp
  python_cpp/gen-cpp/python_cpp_constants.cpp
)
target_link_libraries(deep_3d_descriptor
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  thrift
)

add_library(match_deep_3d_descriptor SHARED
  src/match_deep_3d_descriptor.cpp
)

add_executable(compute_deep_3d_descriptor
  src/compute_deep_3d_descriptor.cpp
)
target_link_libraries(compute_deep_3d_descriptor
  deep_3d_descriptor
)

add_executable(visualize_deep_3d_descriptor_correspondences
  src/visualize_deep_3d_descriptor_correspondences.cpp
)
target_link_libraries(visualize_deep_3d_descriptor_correspondences
  deep_3d_descriptor
  match_deep_3d_descriptor
)

