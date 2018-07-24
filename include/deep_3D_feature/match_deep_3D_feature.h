#define PCL_NO_PRECOMPILE
#ifndef MATCHDEEP3DFEATURE_H
#define MATCHDEEP3DFEATURE_H
#include <iostream>
#include <fstream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/common/norms.h>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/stdcxx.h>
#include "getFeatures.h"
#include "deep_3D_feature.h"
#include <pcl/registration/correspondence_estimation.h>
class MatchDeep3DFeature
{

  private:
    FeatureCloud features_source_;
    FeatureCloud features_target_;
  public:
    MatchDeep3DFeature();
    void setFeatureSource(const FeatureCloud &feature_source_input) {features_source_ = feature_source_input;}
    void setFeatureTarget(const FeatureCloud &feature_target_input) {features_target_ = feature_target_input;}
    void estimateCorrespondences(pcl::Correspondences &correspondences);

};

#endif
