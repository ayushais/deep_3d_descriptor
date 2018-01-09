#include "deep_3D_feature/deep_3D_feature.h"
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;


Deep3DFeature::Deep3DFeature(){}



void Deep3DFeature::get_image_patch(const float min_val,const float max_val,std::vector<cv::Mat>&image_patches_vector)
{
  std::vector<cv::Mat>vecD_initial;
  vecD_initial.resize(keypoints_.points.size());
  #pragma omp parallel for
  for(size_t p = 0; p < keypoints_.points.size();++p)
  {
    IntensityPoint point = keypoints_.points[p];
    Eigen::Vector4f min_vec_big,max_vec_big;
    float min_x = point.x - (nb_radius_/2);
    float max_x = point.x + (nb_radius_/2);

    float min_y = point.y - (nb_radius_/2);
    float max_y = point.y + (nb_radius_/2);

    float min_z = point.z - (nb_radius_/2);
    float max_z = point.z + (nb_radius_/2);
    float voxel_min_y = min_y;
    float voxel_min_z = min_z;

    min_vec_big[0] = min_x;
    min_vec_big[1] = min_y;
    min_vec_big[2] = min_z;
    min_vec_big[3] = 1;

    max_vec_big[0] = max_x;
    max_vec_big[1] = max_y;
    max_vec_big[2] = max_z;
    max_vec_big[3] = 1;
    float max_depth = max_vec_big.lpNorm<2>();
    float min_depth = min_vec_big.lpNorm<2>();
    float step_size = nb_radius_/patch_size;
    std::vector<int>indices;
    pcl::getPointsInBox(cloud_,min_vec_big,max_vec_big,indices);
    IntensityCloud cloud_box;

 ///get the points in the cube
    pcl::copyPointCloud(cloud_,indices,cloud_box);
///if number of points in the cube is too less then descriptor would be
//inaccurate due to lack of local surface
    if(indices.size() < 100)
      continue;

    Eigen::Vector4f min_cloud,max_cloud;

    ///not all voxels in the cube of nb size have points. To avoid iterating
    //over empty voxels estimate a rough dimension of the points inside the
    //neighbourhood cuboid.
    pcl::getMinMax3D(cloud_box,min_cloud,max_cloud);

    ///All the voxels outside the cuboid(min cloud and max cloud) are empty



    cv::Mat image_patch(patch_size,patch_size,CV_32FC3,cv::Scalar(0,0,0));
    int count_detected = 0;
    //int ctr = 0;
    std::stringstream ss;

    ///outerloop for iterating over y direction
    for(size_t i = 1; i <= patch_size;i++)
    {
      float voxel_max_z = voxel_min_z + step_size;
      float voxel_min_y = min_y;

      int count_empty = 0;
      float voxel_max_y_row = voxel_min_y + (step_size * patch_size);
      Eigen::Vector4f min_vec_row(min_x,voxel_min_y,voxel_min_z,1.0);
      Eigen::Vector4f max_vec_row(max_x,voxel_max_y_row,voxel_max_z,1.0);
      IntensityCloud cloud_slice;
      std::vector<int>indices_slice;
      ////outside the cloud cuboid
      if(max_vec_row[2] < min_cloud[2]||min_vec_row[2] > max_cloud[2]||
            max_vec_row[0] < min_cloud[0]||min_vec_row[0] > max_cloud[0]||
            max_vec_row[1] < min_cloud[1]||min_vec_row[1] > max_cloud[1])
      {
        voxel_min_y = voxel_max_y_row;
        voxel_min_z = voxel_max_z;
        continue;

      }
      else
      {
        pcl::getPointsInBox(cloud_box,min_vec_row,max_vec_row,indices_slice);

        ///voxel is empty

        if(indices_slice.empty())
        {
          voxel_min_y = voxel_max_y_row;
          voxel_min_z = voxel_max_z;

          continue;


        }
        else
        {
          pcl::copyPointCloud(cloud_box,indices_slice,cloud_slice);
        }
      }

      ///inner loop for x direction
      for(size_t k = 1; k <=patch_size;k++)
      {

        float voxel_max_y = voxel_min_y + step_size;
        Eigen::Vector4f min_vec(min_x,voxel_min_y,voxel_min_z,1.0);
        Eigen::Vector4f max_vec(max_x,voxel_max_y,voxel_max_z,1.0);

        ////outside the cloud cuboid
        if(max_vec[2] < min_cloud[2]||min_vec[2] > max_cloud[2]||
            max_vec[0] < min_cloud[0]||min_vec[0] > max_cloud[0]||
            max_vec[1] < min_cloud[1]||min_vec[1] > max_cloud[1])
        {
          voxel_min_y = voxel_max_y;

          continue;

        }

        //indices.clear();

        std::vector<int>indices_bbox;
        pcl::getPointsInBox(cloud_slice,min_vec,max_vec,indices_bbox);

        if(indices_bbox.empty())
        {

          count_empty+=1;
          voxel_min_y = voxel_max_y;

          continue;


        }


        float avg_depth = 0.;
        float avg_height = 0.;
        float avg_intensity = 0.;

        ///Estimate the average depth, intensity and height of a point
        for(auto &index:indices_bbox)
        {
          Eigen::Vector3f point_vec = cloud_slice.points[index].getVector3fMap();
          avg_depth += point_vec.lpNorm<2>();
          pcl::PointXYZI point = cloud_.points[indices[indices_slice[index]]];

          float height = (point.z + abs(min_val)) / (max_val + abs(min_val));
          avg_height += height;

          avg_intensity += point.intensity;
        }


        avg_depth/=indices_bbox.size();
        avg_height/=indices_bbox.size();
        avg_intensity/=indices_bbox.size();
/// normalize the depth so that it is defined w.r.t to the neighourhood
//and not with respect to the sensor
        float depth_pixel = (min_depth - avg_depth)/(min_depth - max_depth);

////store the values in the image
        image_patch.at<cv::Vec3f>(63-(i-1),63-(k-1))[2] = depth_pixel;
        image_patch.at<cv::Vec3f>(63-(i-1),63-(k-1))[1] = avg_height;
        image_patch.at<cv::Vec3f>(63-(i-1),63-(k-1))[0] = avg_intensity;

        voxel_min_y = voxel_max_y;
        //ctr+=1;

      }
        voxel_min_z = voxel_max_z;
      }

       vecD_initial[p] = image_patch;

  }

  size_t count_zeros = 0;
  size_t ctr = 0;
  for(auto &image_patch:vecD_initial)
  {
    if(image_patch.rows == 0)///the keypoint didn't have a patch
    {
      count_zeros+=1;

      ctr+=1;
      continue;
    }

    else
    {
      selected_keypoints_.points.push_back(keypoints_.points[ctr]);
      cv::Mat image_patch_8bit;
      cv::convertScaleAbs(image_patch,image_patch_8bit,255,0);
      cv::Mat image_patch_float(64,64,CV_64FC3,cv::Scalar(0.0));
      image_patch_8bit.convertTo(image_patch_float,CV_64F,1.0);
      image_patches_vector.push_back(image_patch_float);

    }

    ctr+=1;
  }




}




void Deep3DFeature::computeFeature(FeatureCloud &features)
{


  std::cout << "Input PointCloud size: " << cloud_.size() << std::endl;
  std::cout << "Input keypoint size: " << keypoints_.size() << std::endl;
  std::cout << "Input neighbourhood size: " << nb_radius_ << std::endl;

  std::cout << "computing surface patches ...." << std::endl;
  float mean_z = 0;
  for (auto point : cloud_.points) {
     mean_z += point.z;
  }

  float sd = 0;

  mean_z /= cloud_.points.size();
  for (auto point : cloud_.points) {
    sd += ((point.z - mean_z) * (point.z - mean_z));
  }

  sd /= cloud_.points.size();

  float min_val = (mean_z - (1.5 * sqrt(sd)));
  float max_val = (mean_z + (1.5 * sqrt(sd)));

  selected_keypoints_.points.clear();
  std::vector<cv::Mat>keypoint_image_patches;
  get_image_patch(min_val,max_val,keypoint_image_patches);
  std::cout << "surface patches computed" << std::endl;
  std::cout << "out of " << keypoints_.size()
    << " keypoints, patches computed for " << selected_keypoints_.size() << " keypoints" << std::endl;



  std::cout << "calling python service for computing features" << std::endl;

  stdcxx::shared_ptr<TSocket> socket(new TSocket("toffifee", 9090));
  stdcxx::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  stdcxx::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  getFeaturesClient client(protocol);
  transport->open();
  size_t ctr = 0;
  std::vector<double>patch_vector (selected_keypoints_.size() * patch_size * patch_size * 3);
  for(auto &image_patch:keypoint_image_patches)
  {
    cv::Mat split_image_patch[3];
    cv::split(image_patch,split_image_patch);
    for(size_t channel = 0; channel < 3; ++channel)
    {
      std::copy(split_image_patch[channel].begin<double>(),split_image_patch[channel].end<double>(),patch_vector.begin() + (ctr * 64 * 64));
      ctr+=1;
    }
  }

  std::vector<double>feature_vector(selected_keypoints_.size() * 256);
  client.returnFeature(feature_vector,patch_vector);
  transport->close();
  std::cout << "feature estimated" << std::endl;

  for(size_t i = 0; i < selected_keypoints_.size() ; ++i)
  {
    DeepFeature256 deep_feature;
    int start_index = i * 256;
    std::copy(feature_vector.begin() + start_index,feature_vector.begin() + start_index + 256,deep_feature.descriptor);
    features.points.push_back(deep_feature);
  }

  cloud_.points.clear();
  keypoints_.points.clear();


  //ctr = 0;
/*  for(auto &feature:features.points)*/
  //{

    //for(size_t i = 0; i < 256 ; ++i)
    //{

      //std::cout << feature.descriptor[i] << "," << feature_vector[ctr] << std::endl;

      //ctr+=1;

      //getchar();


    //}



  //}





  //Eigen::MatrixXd feature_vector_eigen = Eigen::Map<Eigen::MatrixXd>(feature_vector.data(),256,selected_keypoints.size());


/*  std::cout << feature_vector_eigen.rows() << std::endl;*/
  //std::cout << feature_vector_eigen.cols() << std::endl;



  /*for(size_t i = 0; i < selected_keypoints.size() ; ++i)*/
    //features.points.push_back(feature_vector_eigen.col(i));

    //std::cout << feature_vector_eigen.col(0)[i] << "," << feature_vector[i] << std::endl;


    //getchar();






  //std::cout << m1.rows << std::endl;
  //features.resize(selected_keypoints.size());


      //Eigen::MatrixXf feature(1,feature_size);

/*  for(auto &feature_element:feature_vector)*/
  //{



  //}








}
