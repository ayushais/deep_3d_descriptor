#include "deep_3d_descriptor/deep_3d_descriptor.h"

int main(int argc,char **argv)
{
/*  if(argc < 6)*/
  //{
    //std::cerr << "The input is path to the pointcloud, the neighbourhood radius  and sampling radius" << std::endl;
    //return(1);
  /*}*/

  boost::filesystem::path input_path;
  float neighbourhood_radius;
  float sampling_radius;
  if(pcl::console::find_argument(argc, argv, "--path_to_pcd_file") >= 0)
    input_path = argv[pcl::console::find_argument(argc, argv, "--path_to_pcd_file")+1];
  else
  {
    std::cerr << "pointcloud path not given" << std::endl;
    return(1);
  }


  if(pcl::console::find_argument(argc, argv, "--feature_neighborhood_radius") >= 0)
  {
    neighbourhood_radius = atof(argv[pcl::console::find_argument(argc, argv, "--feature_neighborhood_radius")+1]);
  }
  else
  {
    std::cerr << "neighbourhood radius not given" << std::endl;
    return(1);
  }
  if(pcl::console::find_argument(argc, argv, "--sampling_radius") >= 0)
  {
    sampling_radius = atof(argv[pcl::console::find_argument(argc, argv, "--sampling_radius")+1]);
  }
  else
  {
    std::cerr << "sampling radius not given" << std::endl;
    return(1);
  }




  std::string pointcloud_path = input_path.string();
  std::string filename = input_path.filename().string();
  size_t found = filename.find(".pcd");
  if(found == -1)
  {
    std::cerr << "the input file is not a pcd file" << std::endl;
    return(1);

  }

  //prefix to store the output
  filename = filename.substr(0,found);
  pcl::PCDReader reader;
  pcl::PCDWriter writer;
  IntensityCloud::Ptr input_cloud(new IntensityCloud);
  reader.read(pointcloud_path,*input_cloud);
  std::cout << "cloud loaded with " << input_cloud->points.size()  << " number of points" << std::endl;
  if(input_cloud->points.empty())
  {
    std::cerr << "path might be wrong because pointcloud is empty" << std::endl;
    return(1);

  }


////finding keypoints using uniform sampling. Can be replaced by any keypoint detector
  IntensityCloud::Ptr cloud_filter(new IntensityCloud);
  pcl::UniformSampling<IntensityPoint> uniform_sampling;
  uniform_sampling.setInputCloud(input_cloud);
  uniform_sampling.setRadiusSearch(sampling_radius);
  uniform_sampling.filter(*cloud_filter);
  std::cout << "number of keypoins: " << cloud_filter->points.size() << std::endl;
  Deep3DDescriptor deep_feature;
///computing feature descriptors
  deep_feature.setInputCloud(input_cloud);
  deep_feature.setKeypoints(cloud_filter);
  deep_feature.setRadius(neighbourhood_radius);
  FeatureCloud deep_features;
  deep_feature.compute(deep_features);
  std::cout << "total number of features computed are: " << deep_features.points.size() << std::endl;


}
