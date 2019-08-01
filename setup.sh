#!/bin/sh

############# HPATCHES #################
printf "\n \n \n \n____________Downloading hpatches dataset____________ \n\n"
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xvzf hpatches-sequences-release.tar.gz
rm hpatches-sequences-release.tar.gz
echo "Extraction complete!"
#pwd is fancy-keypoints

################ SFOP ##################
printf "\n \n \n \n______________Setting up sfop detector_____________\n \n"
#wget http://www.ipb.uni-bonn.de/html/software/sfop/sfop-0.9.tar.gz 
#tar -xvzf sfop-0.9.tar.gz
#rm sfop-0.9.tar.gz
#cd sfop-0.9
#./configure
cd sfop
mkdir -p build
cd build
cmake ..
make
cd ../..
# rm -R sfop-0.9
echo "sfop installation successful"
#pwd is fancy-keypoints
# if there are comments in the second line of the image, then remove them
# because c_img library cannot read these images 
bash sfop/pre_process.sh

############### SUPERPOINT MODEL #######################
printf "\n \n \n \n ____________Downloading pretrained model for SuperPointNetwork____________ \n \n"
cd superpoint
wget https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth
echo "SuperpointNet: Download complete"  
cd ..
# pwd is fancy-keypoints

################# D2NET MODEL ######################
printf "\n \n \n \n __________________Downloading pretrained model for D2-net_________________ \n \n"
cd d2net
wget https://dsmn.ml/files/d2-net/d2_tf.pth
cd ..
# pwd is fancy-keypoints


################# LIFT MODEL ######################
printf "\n \n \n \n __________________Downloading pretrained model for LIFT_________________ \n \n"
cd lift
mkdir models
cd models
mkdir base configs picc-best
cd base
wget https://github.com/cvlab-epfl/LIFT/raw/master/models/base/new-CNN3-picc-iter-56k.h5 
cd ../configs
wget https://github.com/cvlab-epfl/LIFT/raw/master/models/configs/picc-finetune-nopair.config 
cd ../picc-best
wget https://github.com/cvlab-epfl/LIFT/raw/master/models/picc-best/mean_std.h5
wget https://github.com/cvlab-epfl/LIFT/raw/master/models/picc-best/model.h5 
echo "LIFT: Download complete"
cd ../../../
# pwd is fancy-keypoints
# cd lift/lib/c-code


echo "===> SETUP Complete"
echo "."
echo "."
echo "."
printf "\n \n \n=============================================== \n"
echo "Next: Extract keypoints and descriptors"
echo "Go through README and extract_features.py"
