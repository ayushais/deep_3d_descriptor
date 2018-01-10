#include "deep_3D_feature/deep_3D_feature.h"
#include "deep_3D_feature/match_deep_3D_feature.h"
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
  float sampling_radius_source = atof(argv[2]);
  float sampling_radius_target = atof(argv[4]);
  float neighbourhood_radius = atof(argv[5]);
  size_t use_metric = atoi(argv[6]);
  pcl::PCDReader reader;
  pcl::PCDWriter writer;

  boost::filesystem::path input_path = argv[1];
  string pointcloud_path = input_path.string();
  string filename = input_path.filename().string();
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

  input_path = argv[3];
  pointcloud_path = input_path.string();
  filename = input_path.filename().string();
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
  IntensityCloud keypoints_source;
  pcl::UniformSampling<IntensityPoint> uniform_sampling;
  uniform_sampling.setInputCloud(input_cloud_source);
  uniform_sampling.setRadiusSearch(sampling_radius_source);
  uniform_sampling.filter(keypoints_source);
  cout << "number of keypoins for cloud 1: " << keypoints_source.points.size() << endl;


  IntensityCloud keypoints_target;
  uniform_sampling.setInputCloud(input_cloud_target);
  uniform_sampling.setRadiusSearch(sampling_radius_target);
  uniform_sampling.filter(keypoints_target);
  cout << "number of keypoints for cloud 2: " << keypoints_target.points.size() << endl;


  Deep3DFeature deep_feature;

  deep_feature.setInputCloud(*input_cloud_source);
  deep_feature.setKeypoints(keypoints_source);
  deep_feature.setRadius(neighbourhood_radius);
  FeatureCloud deep_features_source;
  deep_feature.computeFeature(deep_features_source);

  IntensityCloud selected_keypoints_source = deep_feature.getSelectedKeypoints();


  deep_feature.setInputCloud(*input_cloud_target);
  deep_feature.setKeypoints(keypoints_target);
  deep_feature.setRadius(neighbourhood_radius);
  FeatureCloud deep_features_target;
  deep_feature.computeFeature(deep_features_target);
  IntensityCloud selected_keypoints_target = deep_feature.getSelectedKeypoints();

  pcl::Correspondences correspondences;

  if(use_metric == 1)
  {
    cout << "using metric learning for matching features" << endl;
    MatchDeep3DFeature est_deep_correspondences;
    est_deep_correspondences.setFeatureSource(deep_features_source);
    est_deep_correspondences.setFeatureTarget(deep_features_target);
    est_deep_correspondences.estimateCorrespondences(correspondences);
  }

  else
  {
    cout << "using euclidean distance for matching features" << endl;
    pcl::registration::CorrespondenceEstimation<DeepFeature256,DeepFeature256> est;
    est.setInputSource (deep_features_source.makeShared());
    est.setInputTarget (deep_features_target.makeShared());
    est.determineCorrespondences (correspondences);
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("3D Viewer"));

  pcl::visualization::PointCloudColorHandlerCustom<IntensityPoint> purple(input_cloud_source, 51, 0, 102);
  pcl::visualization::PointCloudColorHandlerCustom<IntensityPoint> green(input_cloud_target, 0, 102, 0);
  viewer->addPointCloud<IntensityPoint>(input_cloud_source,green,"input_cloud_source");
  viewer->addPointCloud<IntensityPoint>(input_cloud_target,purple,"input_cloud_target");


  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color_kp_1(selected_keypoints_source.makeShared(), 255, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color_kp_2(selected_keypoints_target.makeShared(), 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZI>(selected_keypoints_source.makeShared(),color_kp_1,"selected_keypoints_source");
  viewer->addPointCloud<pcl::PointXYZI>(selected_keypoints_target.makeShared(),color_kp_2,"selected_keypoints_target");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "selected_keypoints_source");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "selected_keypoints_target");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "input_cloud_source");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "input_cloud_target");
  viewer->setBackgroundColor(255,255,255);



  size_t ctr = 0;
  std::stringstream ss;
  for(auto &corr:correspondences)
  {

    ss.str("");
    ss << "line_" << ctr;
    viewer->addLine(selected_keypoints_source.points[corr.index_query],selected_keypoints_target.points[corr.index_match],255,0,0,ss.str());

    ctr+=1;



  }


  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));

  }






}
