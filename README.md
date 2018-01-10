# deep_3D_feature

Code for the paper "Deep3DFeatures: Learning Local Feature Descriptors for 3D LiDAR Scans".
This package provides the code for training a model for learning and matching
the feature descriptors. We also provide a C++ library for using the learned features descriptor with PCL. 




## Getting Started

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

In the project directory

```
mkdir build
cmake .. 
make

```
In case PCL 1.8 is not found, use -DPCL_DIR variable to specify the path of PCL installation
```
cmake .. -DPCL_DIR:STRING=PATH_TO_PCLConfig.cmake

```

## Training the model

```
python siamese.py --filename --path_to_train_hdf5 --path_to_test_hdf5

```
This command has to be executed in a TensorFlow envionment. 

## Using the learned descriptor with PCL

We provide a service and client API for using the learned feature descriptor with PCL.

All the Thrift related code and the python service file is in the folder python-cpp

The service has to be started within the tensorflow environment
```
python PythonServer.py --model_name

```
We provide two test files, the first one for computing a feature descriptor and 
the second one for matching the descriptors.

For computing feature descriptor 
```
./compute_deep_3D_feature --path_to_pcd_file  --feature_neighborhood_radius --sampling_radius_for_keypoints

```

For matching feafeature descriptor 
```
./visualize_deep_3D_feature_correspondence --path_to_first_pcd_file --sampling_radius_for_first_pcd 
 --path_to_second_pcd_file --sampling_radius_for_second _pcd --feature_neighborhood_radius 
 --use_learned_for_matching_or_use_euclidean_distance(1/0)

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







The package provides the code for the following things:

1. Learning a feature descriptor for 3D LiDAR scans
2. A library for using learned descriptor with PCL


###Files for learning the feature descriptor:


python_scripts/siamese.py: file used for training the deep neural network using TensorFlow

python siamese.py --filename --path_to_train_hdf5_file --path_to_test_hdf5_file



  












