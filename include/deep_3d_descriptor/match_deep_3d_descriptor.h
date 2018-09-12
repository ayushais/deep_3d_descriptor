#define PCL_NO_PRECOMPILE
#ifndef MATCHDEEP3DDESCRIPTOR_H
#define MATCHDEEP3DDESCRIPTOR_H
#include "deep_3d_descriptor/deep_3d_descriptor.h"
class MatchDeep3DDescriptor
{

  private:
    DescriptorCloud features_source_;
    DescriptorCloud features_target_;
  public:
    MatchDeep3DDescriptor();
    void setFeatureSource(const DescriptorCloud &feature_source_input);
    void setFeatureTarget(const DescriptorCloud &feature_target_input);
    void estimateCorrespondences(pcl::Correspondences &correspondences);

};

#endif
