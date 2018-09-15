#define PCL_NO_PRECOMPILE
#ifndef DEEP3DDESCRIPTOR_H
#define DEEP3DDESCRIPTOR_H
#include <iostream>
#include <fstream>
#include <string>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/common/norms.h>
#include <pcl/console/parse.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <thrift/transport/TSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/stdcxx.h>

#include <opencv2/highgui/highgui.hpp>

#include "gen-cpp/get_descriptors.h"

struct DeepDescriptor256
{
  float descriptor[256];
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment


POINT_CLOUD_REGISTER_POINT_STRUCT (DeepDescriptor256,
     (float[256], descriptor, descriptor)
 )

using IntensityPoint = pcl::PointXYZI;
using IntensityCloud = pcl::PointCloud <IntensityPoint>;
using DescriptorCloud = pcl::PointCloud <DeepDescriptor256>;
constexpr int kPatchSize = 64;
constexpr int kDescriptorSize = 256;
class Deep3DDescriptor
{

  private:
    IntensityCloud::Ptr cloud_;
    float nb_radius_;
    IntensityCloud::Ptr keypoints_;
    IntensityCloud::Ptr selected_keypoints_;
    void getImagePatch(std::vector<cv::Mat>&image_patches_vector);

  public:
    Deep3DDescriptor();
    void setInputCloud(const IntensityCloud::Ptr &cloud_input);
    void setRadius(const float nb_radius_input);
    void setKeypoints(const IntensityCloud::Ptr &keypoints_input);
    IntensityCloud getSelectedKeypoints();
    void compute(DescriptorCloud &feature);

};

#endif
