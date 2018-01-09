cmake_minimum_required (VERSION 2.6)

SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  /home/dewan/software/pcl/build/)



project (deep_3D_feature)
set(PCL_DIR "/export/home/dewan/software/pcl/build/")
include_directories("/home/dewan/software/gnuplot-iostream/")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/)


find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
MESSAGE(${PCL_INCLUDE_DIRS})
 

find_package(OpenCV 2.4.8 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})




link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build)

find_package(Boost REQUIRED COMPONENTS system)
FIND_PACKAGE ( Threads REQUIRED )


set(THRIFT_DIR "/usr/local/include/thrift")
include_directories(${THRIFT_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/python-cpp/gen-cpp)
link_directories(/usr/local/lib)


include_directories("/usr/include/eigen3/")

find_package(OpenMP)
if (OPENMP_FOUND)
       set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
SET(CMAKE_CXX_FLAGS " -pthread -fopenmp -std=c++11 -O3 -Wno-deprecated  -DHAVE_INTTYPES_H -DHAVE_NETINET_IN_H")

set(BASE_SOURCE_FILES python-cpp/gen-cpp/getFeatures.cpp python-cpp/gen-cpp/pythonCpp_types.cpp
  python-cpp/gen-cpp/pythonCpp_constants.cpp)
set(CLIENT_FILES src/estimate_input_patch.cpp)

add_library(deep3dfeature SHARED src/deep_3D_feature.cpp ${BASE_SOURCE_FILES})
target_link_libraries(deep3dfeature thrift ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(matchdeep3dfeature SHARED src/match_deep_3D_feature.cpp)
target_link_libraries(matchdeep3dfeature thrift ${PCL_LIBRARIES})

add_executable(compute_deep_3D_feature src/compute_deep_3D_feature.cpp)
target_link_libraries(compute_deep_3D_feature ${PCL_LIBRARIES} deep3dfeature.so)
add_dependencies(compute_deep_3D_feature deep3dfeature.so)


add_executable(visualize_deep_3D_feature_correspondence src/visualize_deep_3D_feature_correspondence.cpp)
target_link_libraries(visualize_deep_3D_feature_correspondence ${PCL_LIBRARIES} deep3dfeature.so matchdeep3dfeature.so)
add_dependencies(visualize_deep_3D_feature_correspondence deep3dfeature.so matchdeep3dfeature.so)





