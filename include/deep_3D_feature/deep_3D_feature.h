#define PCL_NO_PRECOMPILE
#ifndef DEEP3DFEATURE_H
#define DEEP3DFEATURE_H
#include <iostream>
#include <fstream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/stdcxx.h>
#include "getFeatures.h"

struct DeepFeature256
{
  float descriptor[256];
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment


POINT_CLOUD_REGISTER_POINT_STRUCT (DeepFeature256,
     (float[256], descriptor, descriptor)
 )


typedef pcl::PointXYZI IntensityPoint;
typedef pcl::PointCloud <IntensityPoint> IntensityCloud;
typedef pcl::PointCloud<DeepFeature256> FeatureCloud;

const int patch_size = 64;
const int feature_size = 256;
class Deep3DFeature
{

  private:
    IntensityCloud cloud_;
    float nb_radius_;
    IntensityCloud keypoints_;
    IntensityCloud selected_keypoints_;
    void get_image_patch(const float min_val,const float max_val,
        std::vector<cv::Mat>&image_patches_vector);

  public:
    Deep3DFeature();
    void setInputCloud(const IntensityCloud &cloud_input) {cloud_ = cloud_input;}
    void setRadius(const float nb_radius_input) {nb_radius_ = nb_radius_input;}
    void setKeypoints(IntensityCloud &keypoints_input) {keypoints_ = keypoints_input;}
    IntensityCloud getSelectedKeypoints(){return (selected_keypoints_);}
    void computeFeature(FeatureCloud &feature);

};

#endif
