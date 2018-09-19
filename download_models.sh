#bin/sh
mkdir models
cd models
wget deep3d-descriptor.informatik.uni-freiburg.de/models/deep_3d_descriptor_matching.zip
unzip deep_3d_descriptor_matching.zip
mv deep_3d_descriptor_matching/* .
rm -r deep_3d_descriptor_matching/
rm deep_3d_descriptor_matching.zip
wget deep3d-descriptor.informatik.uni-freiburg.de/models/deep_3d_descriptor_hinge_loss.zip
unzip deep_3d_descriptor_hinge_loss.zip
mv deep_3d_descriptor_hinge_loss/* .
rm -r deep_3d_descriptor_hinge_loss/
rm deep_3d_descriptor_hinge_loss.zip
