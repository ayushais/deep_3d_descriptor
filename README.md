# deep_3D_descriptor

Code for the paper "Learning Local Feature Descriptors for 3D LiDAR Scans".
This package provides the code for training a model for learning and matching
the feature descriptors. We also provide a C++ library for using the learned feature descriptor with PCL. 


## Training the Network 
### Prerequisites

* [Tensorflow]:(https://www.tensorflow.org/install/install_linux)

### Dataset
```
./download_data_network.sh

```
This will download the dataset used for training and testing. The data is the format
NxCxHxW and is stored in hdf5 files. The first channel corresponds to the depth value and the second
channel corresponds to the LiDAR intensity value. For every example there is a label and all the examples corresponding
to the same keypoint have same label. These labels are not used directly for training but only to identify examples
belonging to same keypoint. 

### Training the model
```
python train_model.py --model_name  --path_to_training_data  --path_to_testing_data  --batch_size --epochs  --learning_rate  --eta  --growth_rate  --fine_tune_model_name 
```
If the following paramteres are not provided, they will be set to default values
1. batch_size: 32
2. epochs: 5
3. learning_rate: 0.0001
4. eta: 0.0005
5. growth_rate: 4

We recommend the following training procedure. First train the network using the default parameters, then retrain the
network with learning rate set to 0.00001 and weights initialized using the last saved model from the first training. 
The path to the trained model can be set using the paramter --fine_tune_model_name .

```
python train_model.py --model_name another_training_before_release --path_to_training_data ../dataset/training_data.hdf5 --path_to_testing_data ../dataset/testing_data.hdf5 --batch_size 32 --epochs 5 --learning_rate 0.00001 --eta 0.0005 --growth_rate 4 --fine_tune_model_name models/another_training_before_release_110062.ckpt

```
This command has to be executed in a TensorFlow envionment. 

### Testing the model
To test the model we provide the code for calculating the FPR-95 error. The model is tested
on 50,000 positive and negative image patches from the testing data. This script prints the FPR-95 error, plot the curve
between TPR and FPR and stores the data used for plotting the curve.

```
python test_model.py --path_to_saved_model  --path_to_testing_data  --batch_size --epochs  --learning_rate  --eta  --growth_rate  --fine_tune_model_name 

```

## C++ API

git clone command goes here
command for downloading the dataset


### Prerequisites

* [Tensorflow]:(https://www.tensorflow.org/install/install_linux) 

We recommend installing TensorFlow in a virtual environment

* [PCL 1.8]:(https://github.com/PointCloudLibrary/pcl)

* [OpenCV]: (https://github.com/opencv/opencv)

For both C++ and Python

### Installing

Install thrift

```
cd external
./install_thrift.sh

```

In the Tensorflow environment
```
pip install thrift

```


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

## Downloading the test pointcloud

```
./download_test_pcd.sh
```
This will download the test pointcloud files used in alignment experiment in the paper.
The name format for the files is seq_scan_trackID_object.pcd. 
 'seq' corresponds to the sequence number from KITTI tracking benchmark. 'scan' is the scan used from the given
sequence. 'trackID' is the object ID provided by the benchmark. For instance '0011_126_14_object.pcd' and 
'0011_127_14_object.pcd' are the same objects in two consecutive scans.
## Downloading the models

```
./download_models.sh
```
This will download the trained model files. We provide model for a feature descriptor learned simulataneously with
a metric for matching the descriptors and a feature descriptor learned using hinge loss.
'deep_3d_descriptor_matching' contains the learned weights for the descriptor and the metric. 'deep_3d_descriptor_hinge_loss'
 is the learned model for the descriptor trained using hinge loss.



## Using the learned descriptor with PCL

We provide a service and client API for using the learned feature descriptor with PCL.

All the Thrift related code and the python service file is in the folder python_cpp

The service has to be started within the tensorflow environment
```
python python_server.py --model_name --using_hinge_loss

```
We provide two test files, the first one for computing a feature descriptor and 
the second one for matching the descriptors.

For computing feature descriptor 
```
./compute_deep_3d_feature --path_to_pcd_file  --feature_neighborhood_radius --sampling_radius_for_keypoints

```

For visualizing the correspondences between the descriptors and aligning the poinclouds using estimated correspondences.
For aligning the pointclouds, we provide an option of using RANSAC. If RANSAC option is enabled, then the correspondences shown
are from the inlier set estimated by RANSAC. 
```
./visualize_deep_3d_feature_correspondences --path_to_source_pcd_file --sampling_radius_source 
 --path_to_target_pcd_file --sampling_radius_target --feature_neighborhood_radius 
 --use_learned_metric --use_ransac

```

### Example for visualizing the estimated feature correspondences and the aligned pointcloud. The correspondences are estimated using the metric learned by the network


In the Tensorflow environment. python_server.py is in the python_cpp folder

```
python python_server.py --model_name ../models/deep_3d_descriptor_matching.ckpt --use_hinge_loss 0

```

```
./visualize_deep_3d_descriptor_correspondences --path_to_source_pcd_file ../test_pcd/0011_1_2_object.pcd --sampling_radius_source 0.2 --path_to_target_pcd_file ../test_pcd/0011_2_2_object.pcd --sampling_radius_target 0.1 --feature_neighborhood_radius 1.6 --use_learned_metric 1 --use_ransac 0

```

### Example for visualizing the estimated feature correspondences and the aligned pointcloud. The correspondences are estimated using Euclidean distance
 
```

python python_server.py --model_name ../models/deep_3d_descriptor_hinge_loss.ckpt --use_hinge_loss 0

```

```
./visualize_deep_3d_descriptor_correspondences --path_to_source_pcd_file ../test_pcd/0011_1_2_object.pcd --sampling_radius_source 0.2 --path_to_target_pcd_file ../test_pcd/0011_1_2_object.pcd --sampling_radius_target 0.1 --feature_neighborhood_radius 1.6 --use_learned_metric 0 --use_ransac 0

```




Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc




