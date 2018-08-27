#include "deep_3d_descriptor/deep_3d_descriptor.h"
#include "deep_3d_descriptor/match_deep_3d_descriptor.h"
int main(int argc,char **argv)
{
  float sampling_radius_source;
  float sampling_radius_target;
  float neighbourhood_radius;
  int use_metric;
  int use_ransac;
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
    std::cerr << "metric option not given" << std::endl;
    return(1);
  }
  if(pcl::console::find_argument(argc, argv, "--use_ransac") >= 0)
    use_ransac = atoi(argv[pcl::console::find_argument(argc, argv, "--use_ransac")+1]);
  else
  {
    std::cerr << "ransac option not given" << std::endl;
    return(1);
  }

  pcl::PCDReader reader;
  pcl::PCDWriter writer;
  std::string pointcloud_path = input_path_source.string();

  std::string filename = input_path_source.filename().string();
  size_t found = filename.find(".pcd");
  if(found == -1)
  {
    cerr << "the first input file is not a pcd file" << std::endl;
    return(1);

  }

  IntensityCloud::Ptr input_cloud_source(new IntensityCloud);
  reader.read(pointcloud_path,*input_cloud_source);
  std::cout << "first cloud loaded with " << input_cloud_source->points.size()  << " number of points" << std::endl;
  if(input_cloud_source->points.empty())
  {
    cerr << "path might be wrong because the first pointcloud is empty" << std::endl;
    return(1);

  }

  pointcloud_path = input_path_target.string();
  filename = input_path_target.filename().string();
  found = filename.find(".pcd");
  if(found == -1)
  {
    cerr << "the second input file is not a pcd file" << std::endl;
    return(1);

  }
  IntensityCloud::Ptr input_cloud_target(new IntensityCloud);
  reader.read(pointcloud_path,*input_cloud_target);
  std::cout << "target cloud loaded with " << input_cloud_target->points.size()  << " number of points" << std::endl;


  if(input_cloud_target->points.empty())
  {
    cerr << "path might be wrong because the second pointcloud is empty" << std::endl;
    return(1);

  }


  ////prefix to store the output


//////finding keypoints using uniform sampling. Can be replaced by any keypoint detector
  IntensityCloud::Ptr keypoints_source(new IntensityCloud);
  pcl::UniformSampling<IntensityPoint> uniform_sampling;
  uniform_sampling.setInputCloud(input_cloud_source);
  uniform_sampling.setRadiusSearch(sampling_radius_source);
  uniform_sampling.filter(*keypoints_source);
  std::cout << "number of keypoins for cloud 1: " << keypoints_source->points.size() << std::endl;


  IntensityCloud::Ptr keypoints_target(new IntensityCloud);
  uniform_sampling.setInputCloud(input_cloud_target);
  uniform_sampling.setRadiusSearch(sampling_radius_target);
  uniform_sampling.filter(*keypoints_target);
  std::cout << "number of keypoints for cloud 2: " << keypoints_target->points.size() << std::endl;


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
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);

  if(use_metric == 1)
  {
    std::cout << "using metric learning for matching features" << std::endl;
    MatchDeep3DDescriptor est_deep_correspondences;
    est_deep_correspondences.setFeatureSource(deep_features_source);
    est_deep_correspondences.setFeatureTarget(deep_features_target);
    est_deep_correspondences.estimateCorrespondences(*correspondences);
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
            deep_features_target.points[index_target].descriptor,kDescriptorSize);
        if(distance < min_distance)
        {
          min_index = index_target;
          min_distance = distance;

        }




      }

      pcl::Correspondence corr;
      corr.index_query = index_source;
      corr.index_match = min_index;
      correspondences->push_back(corr);


    }
  }
/////plotting correspondence
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_correspondence(
        new pcl::visualization::PCLVisualizer("Correspondences"));

  pcl::visualization::PointCloudColorHandlerCustom<IntensityPoint> green(input_cloud_source, 0, 102, 0);
  pcl::visualization::PointCloudColorHandlerCustom<IntensityPoint> purple(input_cloud_target, 51, 0, 102);
  viewer_correspondence->addPointCloud<IntensityPoint>(input_cloud_source,green,"input_cloud_source");
  viewer_correspondence->addPointCloud<IntensityPoint>(input_cloud_target,purple,"input_cloud_target");
  viewer_correspondence->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "input_cloud_source");
  viewer_correspondence->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "input_cloud_target");
  viewer_correspondence->setBackgroundColor(255,255,255);
  pcl::registration::TransformationEstimationSVD<pcl::PointXYZI,pcl::PointXYZI,float> svd;
  Eigen::Matrix4f transformation;
  if(use_ransac == 1)
  {
    pcl::Correspondences inlier_correspondences;
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZI> ransac_filtering;
    ransac_filtering.setInlierThreshold(0.15);
    ransac_filtering.setMaximumIterations(10000);
    ransac_filtering.setInputTarget(selected_keypoints_target.makeShared());
    ransac_filtering.setInputSource(selected_keypoints_source.makeShared());
    ransac_filtering.setInputCorrespondences(correspondences);
    ransac_filtering.getCorrespondences(inlier_correspondences);
    transformation = ransac_filtering.getBestTransformation();
    size_t ctr = 0;
    std::stringstream ss;
    for(auto &corr:inlier_correspondences)
    {
      ss.str("");
      ss << "line_" << ctr;
      viewer_correspondence->addLine(selected_keypoints_source.points[corr.index_query],selected_keypoints_target.points[corr.index_match],255,0,0,ss.str());
      ctr+=1;
    }


  }
  else
  {
    size_t ctr = 0;
    std::stringstream ss;
    for(auto &corr:*correspondences)
    {
      ss.str("");
      ss << "line_" << ctr;
      viewer_correspondence->addLine(selected_keypoints_source.points[corr.index_query],selected_keypoints_target.points[corr.index_match],255,0,0,ss.str());
      ctr+=1;
    }
    svd.estimateRigidTransformation(selected_keypoints_source,selected_keypoints_target,*correspondences,transformation);
  }

  std::cout << "Estimated Transformation " << std::endl;
  std::cout << transformation << std::endl;
  IntensityCloud::Ptr cloud_aligned(new IntensityCloud);
  pcl::transformPointCloud<pcl::PointXYZI>(*input_cloud_source,*cloud_aligned,transformation);
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_alignment(
        new pcl::visualization::PCLVisualizer("Alignment"));

  pcl::visualization::PointCloudColorHandlerCustom<IntensityPoint> purple_aligned(cloud_aligned, 51, 0, 102);
  viewer_alignment->addPointCloud<IntensityPoint>(cloud_aligned,purple_aligned,"cloud_aligned");
  viewer_alignment->addPointCloud<IntensityPoint>(input_cloud_target,green,"input_cloud_target");
  viewer_alignment->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "cloud_aligned");
  viewer_alignment->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "input_cloud_target");
  viewer_alignment->setBackgroundColor(255,255,255);





  while (!viewer_correspondence->wasStopped() || !viewer_alignment->wasStopped())
  {
    viewer_correspondence->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    viewer_alignment->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }






}
