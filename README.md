# fancy-keypoints
### Project for keypoint detector and feature extractor evaluator (Abhinav).  
In this work, we compare the performance of interest point detectors and 
descriptors. We evaluate the detectors based on their repeatability and coverage.
We also introduce matching ratios as a metric for measuring the performance of
detectors, providing a comprehensive study. Descriptors are compared based on
their ability to correctly estimate homography. The distintiveness and accuracy
of descriptors is qualitatively assessed using the reprojection error of the nearest
neighbor matches.

For more details, please see
* Full Report: [On Evaluation of Interest Point Detectors and Descriptors](https://gitlab.ipb.uni-bonn.de/amilioto/fancy-keypoints/raw/master/report.pdf?inline=false)

# What does this script do?
This script evaluates interest features on the full sequences of [HPatches Dataset](https://github.com/hpatches/hpatches-dataset). The following features are supported as of now:
1. SIFT (`sift`)
2. ORB (`orb`)
3. SFOP (`sfop`) - It has no descriptor
4. SuperPoint (`superpoint`)
5. D2Net (`d2net`)
6. LIFT (`lift`) 

# Dependencies
* Python 3.6+
* OpenCV
* PyTorch
* h5py imageio imagesize matplotlib numpy scipy tqdm
* For using LIFT: Goto this [link](https://github.com/cvlab-epfl/LIFT/blob/master/requirements.txt) and `pip3 install -r requirements.txt` (You may skip Theano.)
* For using SIFT: You must have OpenCV with SIFT support
* CMake 

# How to Run
1. Run `setup.sh`. It will download the HPatches Dataset, setup SFOP and LIFT. It will also get the pre-trained models for LIFT, SuperPointNet and D2-Net.
2. To evaluate the keypoint detectors, you have to extract them first:
```bash
python extract_features.py --features orb
```
3. Now you can run the evaluation scripts for detectors and descriptors
```bash
python evaluate_kp_detector.py --features orb
python evaluate_descr.py --features orb

```
3. The results will be stored in the `results` directory by the name of the features used.
4. To see the result table, go to the `results` directory and run
```bash
python view_kp_results.py --features orb
python view_descr_results.py --features orb
```


<!-- # todos
* Update LIFT in setup.sh for shared-lib -->

