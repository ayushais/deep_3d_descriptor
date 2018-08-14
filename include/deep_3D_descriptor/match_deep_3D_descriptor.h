#define PCL_NO_PRECOMPILE
#ifndef MATCHDEEP3DDESCRIPTOR_H
#define MATCHDEEP3DDESCRIPTOR_H
#include <pcl/common/norms.h>
#include "deep_3D_descriptor.h"
#include <pcl/registration/correspondence_estimation.h>
class MatchDeep3DDescriptor
{

  private:
    FeatureCloud features_source_;
    FeatureCloud features_target_;
  public:
    MatchDeep3DDescriptor();
    void setFeatureSource(const FeatureCloud &feature_source_input);
    void setFeatureTarget(const FeatureCloud &feature_target_input);
    void estimateCorrespondences(pcl::Correspondences &correspondences);

};

#endif
