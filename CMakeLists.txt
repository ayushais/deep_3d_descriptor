cmake_minimum_required (VERSION 2.6)

SET(CMAKE_BUILD_TYPE Release)
project (deep_3d_descriptor)
SET(PCL_DIR "" CACHE STRING "path to PCLConfig.cmake for 1.8")

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
MESSAGE(${PCL_INCLUDE_DIRS})
 
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

find_package(Boost REQUIRED COMPONENTS system)
FIND_PACKAGE ( Threads REQUIRED )


#set(THRIFT_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/external/thrift-0.11.0/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/python_cpp/gen-cpp)
#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/thrift-0.11.0/lib/)

find_package(OpenMP)
if (OPENMP_FOUND)
       set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
SET(CMAKE_CXX_FLAGS " -pthread -fopenmp -std=c++11 -O3 -Wno-deprecated  -DHAVE_INTTYPES_H -DHAVE_NETINET_IN_H")


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build)


set(BASE_SOURCE_FILES python_cpp/gen-cpp/get_descriptors.cpp python_cpp/gen-cpp/python_cpp_types.cpp
    python_cpp/gen-cpp/python_cpp_constants.cpp)
#set(CLIENT_FILES src/estimate_input_patch.cpp)

add_library(deep3ddescriptor SHARED src/deep_3d_descriptor.cpp ${BASE_SOURCE_FILES})
target_link_libraries(deep3ddescriptor thrift ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(matchdeep3ddescriptor SHARED src/match_deep_3d_descriptor.cpp)
target_link_libraries(matchdeep3ddescriptor thrift ${PCL_LIBRARIES})

add_executable(compute_deep_3d_descriptor src/compute_deep_3d_descriptor.cpp)
target_link_libraries(compute_deep_3d_descriptor ${PCL_LIBRARIES} deep3ddescriptor)
add_dependencies(compute_deep_3d_descriptor deep3ddescriptor)


add_executable(visualize_deep_3d_descriptor_correspondences src/visualize_deep_3d_descriptor_correspondences.cpp)
target_link_libraries(visualize_deep_3d_descriptor_correspondences ${PCL_LIBRARIES} deep3ddescriptor matchdeep3ddescriptor)
#add_dependencies(visualize_deep_3D_descriptor_correspondence deep3ddescriptor matchdeep3ddescriptor)





