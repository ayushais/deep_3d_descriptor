#include "deep_3D_feature/deep_3D_feature.h"
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/uniform_sampling.h>

using namespace std;

int main(int argc,char **argv)
{
  if(argc < 3)
  {
    cerr << "The input is path to the pointcloud, the neighbourhood radius  and sampling radius" << endl;
    return(1);
  }
  boost::filesystem::path input_path = argv[1];
  float neighbourhood_radius = atof(argv[2]);
  float sampling_radius = atof(argv[3]);
  string pointcloud_path = input_path.string();
  string filename = input_path.filename().string();
  size_t found = filename.find(".pcd");
  if(found == -1)
  {
    cerr << "the input file is not a pcd file" << endl;
    return(1);

  }

  //prefix to store the output
  filename = filename.substr(0,found);
  pcl::PCDReader reader;
  pcl::PCDWriter writer;
  IntensityCloud::Ptr input_cloud(new IntensityCloud);
  reader.read(pointcloud_path,*input_cloud);
  cout << "cloud loaded with " << input_cloud->points.size()  << " number of points" << endl;
  if(input_cloud->points.empty())
  {
    cerr << "path might be wrong because pointcloud is empty" << endl;
    return(1);

  }


////finding keypoints using uniform sampling. Can be replaced by any keypoint detector
  IntensityCloud cloud_filter;
  pcl::UniformSampling<IntensityPoint> uniform_sampling;
  uniform_sampling.setInputCloud(input_cloud);
  uniform_sampling.setRadiusSearch(sampling_radius);
  uniform_sampling.filter(cloud_filter);
  cout << "number of keypoins: " << cloud_filter.points.size() << endl;
  Deep3DFeature deep_feature;

  deep_feature.setInputCloud(*input_cloud);
  deep_feature.setKeypoints(cloud_filter);
  deep_feature.setRadius(neighbourhood_radius);
  FeatureCloud deep_features;
  deep_feature.computeFeature(deep_features);

  cout << "total number of features computed are: " << deep_features.points.size() <<endl;
/*  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);*/
  //pcl::registration::CorrespondenceEstimation<DeepFeature256,DeepFeature256> est;
  //est.setInputSource (deep_features.makeShared());
  //est.setInputTarget (deep_features.makeShared());
  //est.determineCorrespondences (*correspondences);

  //for(auto &corr:*correspondences)
  //{
    //cout << corr.index_query << "," << corr.index_match << endl;
    //getchar();

  //}








  /*  const size_t patch_width = 64;*/
  /*const size_t patch_height = 64;*/


}
