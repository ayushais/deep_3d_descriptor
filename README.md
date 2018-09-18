# Deep 3D Descriptor

This repository contains code to learn and apply a local feature descriptor for 3D LiDAR scans. We provide the scripts to train the model and a C++ library to interface the learned decriptor with PCL. Training data to learn the model as well as trained models are available.

#### Related Publication

Ayush Dewan, Tim Caselitz, Wolfram Burgard  
**[Learning a Local Feature Descriptor for 3D LiDAR Scans](http://ais.informatik.uni-freiburg.de/publications/papers/dewan18iros.pdf)**  
*IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, Spain, 2018*  

## 1. License

This software is released under GPLv3. If you use it in academic work, please cite:

```
@inproceedings{dewan2018iros,
  author = {Ayush Dewan and  Tim Caselitz and Wolfram Burgard},
  title = {Learning a Local Feature Descriptor for 3D LiDAR Scans},
  booktitle = {Proc.~of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = 2018,
  url = {http://ais.informatik.uni-freiburg.de/publications/papers/dewan18iros.pdf}
}
```


## 2. Training the Network 

### 2.1. Prerequisites

* Tensorflow
* Pyhton 2.7
* H5py

### 2.2. Dataset

```
./download_dataset.sh

```

This will download the dataset used for training and testing. The data is in the format NxCxHxW and is stored in hdf5 files. The first channel corresponds to the depth value and the second channel corresponds to the LiDAR intensity value. For every example there is a label and all the examples corresponding to the same keypoint have same label. These labels are not used directly for training but only to identify examples belonging to same keypoint.

### 2.3. Training the model
All the files required for training and testing the model is in python_scripts folder. To train the model following script has to be executed.

```
python train_model.py 

Parameters
--model_name
--path_to_training_data
--path_to_testing_data
--fine_tune_model_name
--path_to_store_models (default: learned_models/)
--batch_size (default: 32)
--epochs (default: 5)
--learning_rate (default: 0.0001)
--eta (default: 0.0005)
--growth_rate (default: 4)
--number_of_models_stored (default: 2)

```

We recommend the following training procedure. Train the network with the default parameters. Then retrain the network with learning rate set to 0.00001 and weights initialized using the  last saved model from the first training. The path to the trained model can be set using the paramter --fine_tune_model_name.

#### 2.3.1. Example commands for completing the above mentioned training procedure:

```
python train_model.py --model_name  my_model --path_to_training_data ../dataset/training_data.hdf5  --path_to_testing_data  ../dataset/testing_data.hdf5
```

```
python train_model.py --model_name  my_model_retrain --path_to_training_data ../dataset/training_data.hdf5  --path_to_testing_data  ../dataset/testing_data.hdf5 --learning_rate 0.00001 --fine_tune_model_name learned_models/my_model_110062

```

### 2.4. Testing the model
To test the model we provide the code for calculating the FPR-95 error. The model is tested on 50,000 positive and negative image patches from the testing data. This script prints the FPR-95 error, plot the curve between TPR and FPR and stores the data used for plotting the curve.

```
python test_model.py

Parameters
--path_to_saved_model
--path_to_testing_data

```

#### 2.4.1. Example command for testing a trained model
```
python test_model.py --path_to_saved_model learned_models/my_model_retrain_55031  --path_to_testing_data ../dataset/testing_data.hdf5

```


## 3. C++ PCL Interface

### 3.1. Prerequisites

* Tensorflow
* [PCL 1.8] (https://github.com/PointCloudLibrary/pcl)
* [OpenCV] (https://github.com/opencv/opencv)
* [Thrift] (https://thrift.apache.org/download)

Thrift is required for both C++ and Python.

### 3.2. Installing

In the project directory

```
mkdir build
cd build
cmake ..
make
```

In case PCL 1.8 is not found, use -DPCL_DIR variable to specify the path of PCL installation
```
cmake .. -DPCL_DIR:STRING=PATH_TO_PCLConfig.cmake
```

### 3.3. Downloading the test pointcloud

```
./download_test_pcd.sh
```

This will download the test pointcloud files used in alignment experiment in the paper. The name format for the files is seq_scan_trackID_object.pcd. 'seq' corresponds to the sequence number from KITTI tracking benchmark. 'scan' is the scan used from the given sequence. 'trackID' is the object ID provided by the benchmark. For instance '0011_126_14_object.pcd' and '0011_127_14_object.pcd' are the same objects in two consecutive scans.

### 3.4. Downloading the models

```
./download_models.sh
```

This will download the trained model files. We provide model for a feature descriptor learned simulataneously with a metric for matching the descriptors and a feature descriptor learned using hinge loss. 'deep_3d_descriptor_matching' contains the learned weights for the descriptor and the metric. 'deep_3d_descriptor_hinge_loss' is the learned model for the descriptor trained using hinge loss.

### 3.5. Using the learned descriptor with PCL

We provide a service and client API for using the learned feature descriptor with PCL.

All the Thrift related code and the python service file is in the folder python_cpp.

The service has to be started within the tensorflow environment
```
python python_server.py

Parameters
--model_name
--using_hinge_loss

```

We provide two test files, the first one for computing a feature descriptor and the second one for matching the descriptors.

For computing feature descriptor

```
./compute_deep_3d_feature

Parameters
--path_to_pcd_file
--feature_neighborhood_radius
--sampling_radius

```

For visualizing the correspondences between the descriptors and aligning the poinclouds using estimated correspondences. For aligning the pointclouds, we provide an option of using RANSAC. If RANSAC option is enabled, then the correspondences shown are from the inlier set estimated by RANSAC. 

```
./visualize_deep_3d_feature_correspondences

Parameters
--path_to_source_pcd_file
--sampling_radius_source
--path_to_target_pcd_file
--sampling_radius_target
--feature_neighborhood_radius
--use_learned_metric
--use_ransac

```

#### 3.5.1. Example for visualizing the estimated feature correspondences and the aligned pointcloud. The correspondences are estimated using the metric learned by the network

In the Tensorflow environment. python_server.py is in the python_cpp folder

```
python python_server.py --model_name ../models/deep_3d_descriptor_matching --use_hinge_loss 0

```

```
./visualize_deep_3d_descriptor_correspondences --path_to_source_pcd_file ../test_pcd/0011_1_2_object.pcd --sampling_radius_source 0.2 --path_to_target_pcd_file ../test_pcd/0011_2_2_object.pcd --sampling_radius_target 0.1 --feature_neighborhood_radius 1.6 --use_learned_metric 1 --use_ransac 0

```
Matched Keypoints             |Aligned Scans
:-------------------------:|:-------------------------:
![](http://deep3d-descriptor.informatik.uni-freiburg.de/corr_metric.png)  |  ![](http://deep3d-descriptor.informatik.uni-freiburg.de/aligned_metric.png)

#### 3.5.2. Example for visualizing the estimated feature correspondences and the aligned pointcloud. The correspondences are estimated using Euclidean distance

```
python python_server.py --model_name ../models/deep_3d_descriptor_hinge_loss --use_hinge_loss 1

```

```
./visualize_deep_3d_descriptor_correspondences --path_to_source_pcd_file ../test_pcd/0011_1_2_object.pcd --sampling_radius_source 0.2 --path_to_target_pcd_file ../test_pcd/0011_2_2_object.pcd --sampling_radius_target 0.1 --feature_neighborhood_radius 1.6 --use_learned_metric 0 --use_ransac 0

```
Matched Keypoints             |Aligned Scans
:-------------------------:|:-------------------------:
![](http://deep3d-descriptor.informatik.uni-freiburg.de/corr_hinge.png)  |  ![](http://deep3d-descriptor.informatik.uni-freiburg.de/aligned_hinge.png)


