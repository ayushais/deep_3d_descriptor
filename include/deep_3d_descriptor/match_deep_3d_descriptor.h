/*
 * This file is part of deep_3d_descriptor.
 *
 * Copyright (C) 2018 Ayush Dewan (University of Freiburg)
 *
 * deep_3d_descriptor is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deep_3d_descriptor is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with deep_3d_descriptor.  If not, see <https://www.gnu.org/licenses/>.
*/

#define PCL_NO_PRECOMPILE
#ifndef MATCHDEEP3DDESCRIPTOR_H
#define MATCHDEEP3DDESCRIPTOR_H
#include "deep_3d_descriptor/deep_3d_descriptor.h"
class MatchDeep3DDescriptor {
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
