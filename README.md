Package for the paper "Deep3DFeatures: Learning Local Feature Descriptors for 3D LiDAR Scans"

The package provides the code for the following things:

1. Learning a feature descriptor for 3D LiDAR scans
2. A library for using learned descriptor with PCL


###Files for learning the feature descriptor:


python_scripts/siamese.py: file used for training the deep neural network using TensorFlow
python siamese.py --filename --path_to_train_hdf5_file --path_to_test_hdf5_file



  












