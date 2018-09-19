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

#include "deep_3d_descriptor/match_deep_3d_descriptor.h"
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

MatchDeep3DDescriptor::MatchDeep3DDescriptor() {}

void MatchDeep3DDescriptor::setFeatureSource(
    const DescriptorCloud &feature_source_input) {
  features_source_ = feature_source_input;
}
void MatchDeep3DDescriptor::setFeatureTarget(
    const DescriptorCloud &feature_target_input) {
  features_target_ = feature_target_input;
}

void MatchDeep3DDescriptor::estimateCorrespondences(
    pcl::Correspondences &correspondences) {
  std::cout << "matching features using learned metric" << std::endl;

  std::cout << "source feature size: " << features_source_.points.size()
            << " and target feature size: " << features_target_.points.size()
            << std::endl;

  std::vector<double> features_source_flattened(features_source_.points.size() *
                                                kDescriptorSize);
  std::vector<double> features_target_flattened(features_target_.points.size() *
                                                kDescriptorSize);

  size_t ctr = 0;
  for (auto &feature : features_source_.points) {
    size_t start_index = ctr * kDescriptorSize;
    std::copy(feature.descriptor, feature.descriptor + kDescriptorSize,
              features_source_flattened.begin() + start_index);
    ctr += 1;
  }

  ctr = 0;
  for (auto &feature : features_target_.points) {
    size_t start_index = ctr * kDescriptorSize;
    std::copy(feature.descriptor, feature.descriptor + kDescriptorSize,
              features_target_flattened.begin() + start_index);
    ctr += 1;
  }

  std::cout << "calling python service for matching features" << std::endl;
  stdcxx::shared_ptr<TSocket> socket(new TSocket("localhost", 9090));
  stdcxx::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  stdcxx::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  get_descriptorsClient client(protocol);
  transport->open();
  std::vector<int> matched_features(features_source_.size());
  client.match_descriptors(matched_features, features_source_flattened,
                           features_target_flattened);
  transport->close();

  ctr = 0;

  for (auto &index_match : matched_features) {
    pcl::Correspondence corr;
    corr.index_query = ctr;
    corr.index_match = index_match;
    correspondences.push_back(corr);
    ctr += 1;
  }

  // std::cout << corr << std::endl;
  std::cout << "correspondences estimated" << std::endl;
}
