import argparse
import os
import time
import numpy as np
import cv2
import pickle
# selectively import other
# packages as needed inside
# functions

from utils import health_check
from utils import homography_loader
from utils import get_overlapping_kp
from utils import get_dist
from utils import evaluate_matches
from utils import get_coverage



# Global variables
input_dir = None
results_dir = None
cache_wg = []
cache_coverage = []

def save_cache_wg(distance, threshold):
    file_path = results_dir + '/kp_eval_' + distance + '_th' + str(threshold)
    with open(file_path, 'wb') as f:
        pickle.dump(cache_wg, f)

def save_cache_coverage():
    file_path = results_dir + '/kp_eval_coverage'
    with open(file_path, 'wb') as f:
        pickle.dump(cache_coverage, f)


class ImagePair(object):
    """ Class loads a pair of images
    and homography 
    """
    def __init__(self, sequence, kp_qry, kp_trg, qry_img_shape, trg_img_shape, homography_name):
        self.sequence = sequence
        self.kp_qry = kp_qry
        self.kp_trg = kp_trg
        self.qry_img_shape = qry_img_shape
        self.trg_img_shape = trg_img_shape
        self.homography_name = homography_name
        source_dir = input_dir + '/' + sequence + '/'
        self.homography_path = source_dir + '/' + homography_name


    def evaluate_pair(self, threshold, distance):    #, visualize):
        """ Funtion to load the overlapping keypoints
        and check the number of matches using 
        adjacency matrix
        """
        H = np.loadtxt(self.homography_path)
        point_qry, point_trg, point_qry_proj_on_trg, point_trg_proj_on_qry = get_overlapping_kp(kp_qry, kp_trg, H, qry_img_shape, trg_img_shape)

        point_qry_len = len(point_qry)
        point_trg_len = len(point_trg)
        # Get the distance matrix
        if point_qry_len == 0 or point_trg_len == 0:
            dist_in_trg_img = np.array([])
            dist_in_qry_img = np.array([])
        else:
            dist_in_trg_img = get_dist(point_qry_proj_on_trg, point_trg, distance)
            dist_in_qry_img = get_dist(point_trg_proj_on_qry, point_qry, distance)
        eval_results = evaluate_matches(dist_in_trg_img, dist_in_qry_img, point_qry_len, point_trg_len, threshold)

        cache_wg.append((self.sequence, self.homography_name, distance, threshold, eval_results))





if __name__ == '__main__':
    start = time.time()
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluation of keypoint detectors.')

    parser.add_argument(
        '--input_dir', type=str, default='hpatches-sequences-release',
        help='Path to the input directory (must follow hpatches directory structure)'
    )    
    parser.add_argument(
        '--features', type=str, default='sift',
        help='Detector to evaluate: sift, orb, sfop, superpoint, d2net'
    )
    parser.add_argument(
        '--distance', type=str, default='euclidean',
        help='Distance metric for reprojection errors: euclidean, cityblock, mahalanobis'
    )
    parser.add_argument(
        '--threshold', type=float, default=1.0,
        help='Threshold reprojection error for counting a match (default=1.0 pixels)'
    )
    
    options = parser.parse_args()
    print(options)

    health_check (options.input_dir, options.features, options.distance)

    path_to_kp = 'results/' + options.features + '/features'

    ############## directory to store results #####################
    results_dir = 'results/' + options.features
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ############## GLOBAL VARIABLE: input directory ###############
    input_dir = options.input_dir


    ################### Load the saved results #####################
    with open(path_to_kp, 'rb') as f:
        keypoint_list = pickle.load(f)
    # print('The pickle file is a list with each item storing:')
    # print('(sequence[v_boat], img_name[1.ppm], img_shape[(680, 850)], keypoints[  array([[x1,y1], [x2,y2], ...])  ])')
    no_of_sequences = int(len(keypoint_list)/6)

    ################### [1] Evaluate keypoints [Wolfgang Stuff] #################
    print('==================== TASK 1 ==================== ')
    print('============ EVALUATE METRICS BY WG =============')
    for i in range(no_of_sequences):
        sequence = keypoint_list[6*i][0]
        source_dir = input_dir + '/' + sequence + '/'
        homography_name = homography_loader()

        print('Sequence In Progress: [[%d] out of %d] ...' %(i+1, no_of_sequences), sequence)

        ############ Run the test for each query target pair #############
        for outer in range(5):
            print('____[[%d]/%d]Image pair in progress: ' %(i+1, no_of_sequences), sequence, homography_name[outer])

            qry_img_path = source_dir + keypoint_list[6*i][1]
            trg_img_path = source_dir + keypoint_list[6*i+1+outer][1]

            qry_img_shape = keypoint_list[6*i][2]
            trg_img_shape = keypoint_list[6*i+1+outer][2]

            kp_qry = keypoint_list[6*i][3]
            kp_trg = keypoint_list[6*i+1+outer][3]

            print(kp_qry.dtype) # this is correct

            ip = ImagePair(sequence, kp_qry, kp_trg, qry_img_shape, trg_img_shape, homography_name[outer])
            ip.evaluate_pair(options.threshold, options.distance)

    print('Writing cached results to disk: ', results_dir)
    save_cache_wg(options.distance, options.threshold)
    print('Results saved!')
    print('The pickle file is a list with each item storing:')
    print('(sequence[v_boat], homography_name[H_1_2], distance[euclidean], threshold[1.0], eval_results[see comments for its contents]) ')
    '''The contents of eval_results are:
    1. kp in qry img in the overlapping region
    2. kp in trg img in the overlapping region
    3. no of unique matches
    4. no of spurious kp in qry img
    5. no of spurious kp in trg img
    6. no of multiple matches
    7. unique match ratio
    8. spurious ratio for qry img
    9. spurious ratio for trg img
    10. multiple match ratio
    11. repeatability score
    '''

    ################# [2] Get coverage of keypoints ########################
    print('==================== TASK 2 ==================== ')
    print('============= COVERAGE IN THE IMAGE =============')
    progress = 1
    len_kp = len(keypoint_list)
    for item in keypoint_list:
        print(progress, 'out of', len_kp, item[0], item[1])
        cache_coverage.append(get_coverage(item))
        progress = progress + 1
    print('Writing coverage results to disk: ', results_dir)
    save_cache_coverage()
    print('Results saved!')
    print('The pickle file is a list with each item storing:')
    print('(sequence[v_boat], img_name[1.ppm], img_shape[(680,850)], no_of_kp[500], hm, hm_normalized) ')
    

    end = time.time()
    print('Total time required in min: %.4f' %((end-start)/60))