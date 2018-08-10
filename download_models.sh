#bin/sh
mkdir models
cd models
wget deep3d-descriptor.informatik.uni-freiburg.de/models/deep_3d_descriptor_matching.zip
unzip deep_3d_descriptor_matching.zip
rm deep_3d_descriptor_matching.zip
wget deep3d-descriptor.informatik.uni-freiburg.de/models/deep_3d_descriptor_hinge_loss.zip
unzip deep_3d_descriptor_hinge_loss.zip
rm deep_3d_descriptor_hinge_loss.zip
