#include "deep_3D_descriptor/deep_3D_descriptor.h"
#include "deep_3D_descriptor/match_deep_3D_descriptor.h"
#include <pcl/filters/uniform_sampling.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>
using namespace std;

int main(int argc,char **argv)
{
  if(argc < 5)
  {
    cerr << "The input is path to the first pointcloud, sampling radius for the first pointcloud" <<
      " path to the second pointcloud, sampli radius for the second poincloud, neighourhood radius and " <<
      " and 1/0 for using metric learning for matching or match features using euclidean distance" << endl;


    return(1);
  }
  float sampling_radius_source;
  float sampling_radius_target;
  float neighbourhood_radius;
  int use_metric;
  boost::filesystem::path input_path_source;
  boost::filesystem::path input_path_target;


  ///source pcd
  if(pcl::console::find_argument(argc, argv, "--path_to_source_pcd_file") >= 0)
    input_path_source = argv[pcl::console::find_argument(argc, argv, "--path_to_source_pcd_file")+1];
  else
  {
    std::cerr << "pointcloud path for source pcd not given" << std::endl;
    return(1);
  }

  //source sampling radius
  if(pcl::console::find_argument(argc, argv, "--sampling_radius_source") >= 0)
    sampling_radius_source = atof(argv[pcl::console::find_argument(argc, argv, "--sampling_radius_source")+1]);
  else
  {
    std::cerr << "sampling radius for source not given" << std::endl;
    return(1);
  }


///target pcd
  if(pcl::console::find_argument(argc, argv, "--path_to_target_pcd_file") >= 0)
    input_path_target = argv[pcl::console::find_argument(argc, argv, "--path_to_target_pcd_file")+1];
  else
  {
    std::cerr << "pointcloud path for target pcd not given" << std::endl;
    return(1);
  }

  ///target sampling radius
  if(pcl::console::find_argument(argc, argv, "--sampling_radius_target") >= 0)
    sampling_radius_target = atof(argv[pcl::console::find_argument(argc, argv, "--sampling_radius_target")+1]);
  else
  {
    std::cerr << "sampling radius for target not given" << std::endl;
    return(1);
  }

  ////feature neighborhood radius
  if(pcl::console::find_argument(argc, argv, "--feature_neighborhood_radius") >= 0)
    neighbourhood_radius = atof(argv[pcl::console::find_argument(argc, argv, "--feature_neighborhood_radius")+1]);
  else
  {
    std::cerr << "neighbourhood radius not given" << std::endl;
    return(1);
  }

  // metric choice
  if(pcl::console::find_argument(argc, argv, "--use_learned_metric") >= 0)
    use_metric = atoi(argv[pcl::console::find_argument(argc, argv, "--use_learned_metric")+1]);
  else
  {
    std::cerr << "metric choice not given" << std::endl;
    return(1);
  }

  pcl::PCDReader reader;
  pcl::PCDWriter writer;
  string pointcloud_path = input_path_source.string();
  string filename = input_path_source.filename().string();
  size_t found = filename.find(".pcd");
  if(found == -1)
  {
    cerr << "the first input file is not a pcd file" << endl;
    return(1);

  }

  IntensityCloud::Ptr input_cloud_source(new IntensityCloud);
  reader.read(pointcloud_path,*input_cloud_source);
  cout << "first cloud loaded with " << input_cloud_source->points.size()  << " number of points" << endl;
  if(input_cloud_source->points.empty())
  {
    cerr << "path might be wrong because the first pointcloud is empty" << endl;
    return(1);

  }

  pointcloud_path = input_path_target.string();
  filename = input_path_target.filename().string();
  found = filename.find(".pcd");
  if(found == -1)
  {
    cerr << "the second input file is not a pcd file" << endl;
    return(1);

  }
  IntensityCloud::Ptr input_cloud_target(new IntensityCloud);
  reader.read(pointcloud_path,*input_cloud_target);
  cout << "target cloud loaded with " << input_cloud_target->points.size()  << " number of points" << endl;


  if(input_cloud_target->points.empty())
  {
    cerr << "path might be wrong because the second pointcloud is empty" << endl;
    return(1);

  }


  ////prefix to store the output


//////finding keypoints using uniform sampling. Can be replaced by any keypoint detector
  IntensityCloud::Ptr keypoints_source(new IntensityCloud);
  pcl::UniformSampling<IntensityPoint> uniform_sampling;
  uniform_sampling.setInputCloud(input_cloud_source);
  uniform_sampling.setRadiusSearch(sampling_radius_source);
  uniform_sampling.filter(*keypoints_source);
  cout << "number of keypoins for cloud 1: " << keypoints_source->points.size() << endl;


  IntensityCloud::Ptr keypoints_target(new IntensityCloud);
  uniform_sampling.setInputCloud(input_cloud_target);
  uniform_sampling.setRadiusSearch(sampling_radius_target);
  uniform_sampling.filter(*keypoints_target);
  cout << "number of keypoints for cloud 2: " << keypoints_target->points.size() << endl;


  Deep3DDescriptor deep_feature;

  deep_feature.setInputCloud(input_cloud_source);
  deep_feature.setKeypoints(keypoints_source);
  deep_feature.setRadius(neighbourhood_radius);
  FeatureCloud deep_features_source;
  deep_feature.compute(deep_features_source);

  IntensityCloud selected_keypoints_source = deep_feature.getSelectedKeypoints();


  deep_feature.setInputCloud(input_cloud_target);
  deep_feature.setKeypoints(keypoints_target);
  deep_feature.setRadius(neighbourhood_radius);
  FeatureCloud deep_features_target;
  deep_feature.compute(deep_features_target);
  IntensityCloud selected_keypoints_target = deep_feature.getSelectedKeypoints();
  pcl::Correspondences correspondences;

  if(use_metric == 1)
  {
    cout << "using metric learning for matching features" << endl;
    MatchDeep3DDescriptor est_deep_correspondences;
    est_deep_correspondences.setFeatureSource(deep_features_source);
    est_deep_correspondences.setFeatureTarget(deep_features_target);
    est_deep_correspondences.estimateCorrespondences(correspondences);
  }

  else
  {


    std::cout << "using Euclidean metric" << std::endl;
    for(size_t index_source = 0; index_source < deep_features_source.points.size(); ++index_source)
    {
      float min_distance = std::numeric_limits<float>::max();
      int min_index = -1;
      for(size_t index_target = 0; index_target < deep_features_target.points.size(); ++index_target)
      {


        float distance = pcl::L2_Norm(deep_features_source.points[index_source].descriptor,
            deep_features_target.points[index_target].descriptor,256);
        if(distance < min_distance)
        {
          min_index = index_target;
          min_distance = distance;

        }


/*        cout <<  << endl;*/

        /*getchar();*/



      }

      pcl::Correspondence corr;
      corr.index_query = index_source;
      corr.index_match = min_index;
      correspondences.push_back(corr);


    }
/*    cout << "using euclidean distance for matching features" << endl;*/
    //pcl::registration::CorrespondenceEstimation<DeepFeature256,DeepFeature256> est;
    //est.setInputSource (deep_features_source.makeShared());
    //est.setInputTarget (deep_features_target.makeShared());
    /*est.determineCorrespondences (correspondences);*/
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("3D Viewer"));



  std::cout << input_cloud_source->points.size() << endl;
  pcl::visualization::PointCloudColorHandlerCustom<IntensityPoint> purple(input_cloud_source, 51, 0, 102);
  pcl::visualization::PointCloudColorHandlerCustom<IntensityPoint> green(input_cloud_target, 0, 102, 0);
  viewer->addPointCloud<IntensityPoint>(input_cloud_source,green,"input_cloud_source");
  viewer->addPointCloud<IntensityPoint>(input_cloud_target,purple,"input_cloud_target");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "input_cloud_source");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "input_cloud_target");
  viewer->setBackgroundColor(255,255,255);



  size_t ctr = 0;
  std::stringstream ss;
  for(auto &corr:correspondences)
  {

    ss.str("");
    ss << "line_" << ctr;
/*    cout << corr.index_query << "," << corr.index_match << endl;*/
    /*getchar();*/
    viewer->addLine(selected_keypoints_source.points[corr.index_query],selected_keypoints_target.points[corr.index_match],255,0,0,ss.str());

    ctr+=1;



  }


  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));

  }






}
