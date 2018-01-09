cmake_minimum_required (VERSION 2.6)

SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  /home/dewan/software/pcl/build/)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

message(${CMAKE_MODULE_PATH})

project (deep_3D_feature)

set(PCL_DIR "/export/home/dewan/software/pcl/build/")
include_directories("/home/dewan/software/gnuplot-iostream/")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/)
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})
MESSAGE(${Caffe_INCLUDE_DIRS})



find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
MESSAGE(${PCL_INCLUDE_DIRS})
 

find_package(OpenCV 2.4.8 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})




include_directories("/home/dewan/code/utils/")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/semantic_segmentation/)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build)
include_directories("/opt/ros/indigo/include/tf2/LinearMath/")
#find_package(lib 1.2.1 REQUIRED)



#link_directories("/home/dewan/code/sceneflow/build/")
	
find_package(Boost REQUIRED COMPONENTS system)
FIND_PACKAGE ( Threads REQUIRED )

find_package(Armadillo 4.2 REQUIRED)
MESSAGE(${ARMADILLO_INCLUDE_DIRS})
MESSAGE(${ARMADILLO_LIBRARIES})

set(THRIFT_DIR "/usr/local/include/thrift")
include_directories(${THRIFT_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/python-cpp/gen-cpp)
link_directories(/usr/local/lib)








include_directories("/usr/include/eigen3/")

include_directories("/home/dewan/software/ImgProj/")
link_directories("/home/dewan/software/ImgProj/build")
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





