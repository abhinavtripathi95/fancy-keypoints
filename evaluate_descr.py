import numpy as np 
import time
import os
import argparse
import pickle
import cv2
from scipy.spatial.distance import cdist

from utils import homography_loader


# define some GLOBAL vars 
results_dir = None
input_dir = None
cache = []

def save_cache():
    file_path = results_dir + '/descr_eval_homography'
    with open(file_path, 'wb') as f:
        pickle.dump(cache, f)

def estimate_homography(kp_qry, descr_qry, kp_trg, descr_trg):
    # I could not get the FLANN of OpenCV to work, 
    # so I implemented my own version of a matcher
    # Find the best and second best match. If the ratio
    # test gives a value less than 0.7, then the match 
    # is 'good'

    qry_len = len(descr_qry)
    dist_descr = cdist(descr_qry, descr_trg, 'euclidean')
    print('dist_descr.dtype',dist_descr.dtype)

    min_idx1 = np.argmin(dist_descr, axis = 1)              # indices of the closest descriptors
    min_dist1 = dist_descr[np.arange(qry_len), min_idx1]

    dist_descr[np.arange(qry_len), min_idx1] = float('inf')

    min_idx2 = np.argmin(dist_descr, axis = 1)              # indices of the second best match
    min_dist2 = dist_descr[np.arange(qry_len), min_idx2]

    idx1 = np.arange(qry_len)
    all = np.c_[idx1, min_idx1]
    good = all[min_dist1 < 0.7*min_dist2]

    # I have found out that at least 4 good matches are required 
    # in order to estimate homography. If the number of good matches 
    # is less than 4, cv2.find_homography returns a None object
    # good = good[0:4,:]
    print('good', good.shape)
    if (good.shape[0] < 4):
        M = None

    else:
        src_pts = np.float32([ kp_qry[m[0]] for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_trg[m[1]] for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    print('H_est',M)
    return M


def homography_error(H, H_est, qry_img_shape):
    # We will estimate how good the homography estimation is by
    # checking how close the 4 corners of the query image are
    # after projecting them on the target image

    # Define c1, c2, c3, c4 as (x,y,1) homogenous coordinates

    c1 = np.array([0,0,1])
    c2 = np.array([0, qry_img_shape[0], 1])
    c3 = np.array([qry_img_shape[1], 0, 1])
    c4 = np.array([qry_img_shape[1], qry_img_shape[0], 1])

    err = np.zeros(4)
    c1_est = np.dot(H_est, c1)
    c1_gt = np.dot(H, c1)
    c1_est = c1_est/c1_est[-1]
    c1_gt = c1_gt/c1_gt[-1]
    err[0] = np.linalg.norm(c1_est - c1_gt)

    c2_est = np.dot(H_est, c2)
    c2_gt = np.dot(H, c2)
    c2_est = c2_est/c2_est[-1]
    c2_gt = c2_gt/c2_gt[-1]
    err[1] = np.linalg.norm(c2_est - c2_gt)

    c3_est = np.dot(H_est, c3)
    c3_gt = np.dot(H, c3)
    c3_est = c3_est/c3_est[-1]
    c3_gt = c3_gt/c3_gt[-1]
    err[2] = np.linalg.norm(c3_est - c3_gt)

    c4_est = np.dot(H_est, c4)
    c4_gt = np.dot(H, c4)
    c4_est = c4_est/c4_est[-1]
    c4_gt = c4_gt/c4_gt[-1]
    err[3] = np.linalg.norm(c4_est - c4_gt)

    error = np.sum(err)
    error = error/4
    return error



if __name__ == '__main__':
    start = time.time()
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate features for homography estimation.')

    parser.add_argument(
        '--input_dir', type=str, default='hpatches-sequences-release',
        help='Path to the input directory (must follow hpatches directory structure)'
    )
    parser.add_argument(
        '--features', type=str, default='sift',
        help='Features to evaluate: sift, orb, superpoint, d2net'
    )
    
    options = parser.parse_args()
    print(options)


    ############## directory to store results #####################
    results_dir = 'results/' + options.features
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ############## GLOBAL VARIABLE: input directory ###############
    input_dir = options.input_dir

    ################### Load the saved results #####################
    path_to_features = 'results/' + options.features + '/features'

    with open(path_to_features, 'rb') as f:
        feature_list = pickle.load(f)

    
    no_of_sequences = int(len(feature_list)/6)
    ################### [1] Evaluate descriptors [Homography] #################
    print('================================ TASK 1 ================================= ')
    print('============ EVALUATE DESCRIPTORS [by homography estimation] =============')

    for i in range(no_of_sequences):
        sequence = feature_list[6*i][0]
        source_dir = input_dir + '/' + sequence + '/'
        homography_name = homography_loader()

        print('Sequence In Progress: [[%d] out of %d] ...' %(i+1, no_of_sequences), sequence)

        ############ Run the test for each query target pair #############
        for outer in range(5):
            print('____[[%d]/%d]Image pair in progress: ' %(i+1, no_of_sequences), sequence, homography_name[outer])

            kp_qry = feature_list[6*i][3]
            kp_trg = feature_list[6*i+1+outer][3]
            
            qry_img_shape = feature_list[6*i][2]
            
            descr_qry = feature_list[6*i][4]
            descr_trg = feature_list[6*i+1+outer][4]

            print('descr.dtype', descr_qry.dtype)

            H_est = estimate_homography(kp_qry, descr_qry, kp_trg, descr_trg)

            if H_est is None:
                h_error = None
            else:
                homography_path = source_dir + '/' + homography_name[outer]
                H = np.loadtxt(homography_path)
                h_error = homography_error(H, H_est, qry_img_shape)
            cache.append((sequence, homography_name[outer], h_error))
    
    save_cache()
