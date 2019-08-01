import argparse
import os
import time
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


from utils import homography_loader
from evaluate_descr import homography_error


# GLOBAL VARIABLES
input_dir = None
less_than_1_px_total = 0        # required for task 1
distribution = None             # required for task 2
task3_h_results = []            # required for task 3
Gcorr = np.zeros(20)             # # required for task 4
Gincorr = np.zeros(20)           # # required for task 4
task5_h_results = []            # requried for task 5

# TASK 1
def check_less_than_one(keypoints):
    kp_dist = cdist(keypoints, keypoints, 'euclidean')
    no_less_than_1 = len(np.where(kp_dist < 1)[0])

    # But remove the diagonal zeros
    no_less_than_1 = no_less_than_1 - len(kp_dist)

    # But remove half because of symmetry
    no_less_than_1 = no_less_than_1/2
    return no_less_than_1

# TASK 2
def get_distribution_of_reprojection_error_of_matches(kp_qry, kp_trg, descr_qry, descr_trg, H):
    dist_descr = cdist(descr_qry, descr_trg, 'euclidean')
    qry_len = len(descr_qry)
    min_idx1 = np.argmin(dist_descr, axis = 1)              # indices of the closest descriptors
    kp_trg_match = kp_trg[min_idx1, :]
    # print(kp_trg_match.shape)

    kp_qry_homogenous = np.c_[kp_qry, np.ones(len(kp_qry))]
    hx = np.dot(H,kp_qry_homogenous.T)
    z = hx[-1,:]
    hx = hx/z[None,:]
    # Convert homogenous coordinates into non-homogenous
    hx = hx.T
    # print('hx.shape', hx.shape)
    kp_qry_proj = hx[:,0:2]
    # print('kp_qry_proj.shape', kp_qry_proj.shape)

    # Now we can find out the actual distances between qry and trg matches
    # print(kp_qry_proj - kp_trg_match)
    required_dist = np.linalg.norm(kp_qry_proj - kp_trg_match, axis=1)
    print('required_dist', required_dist.shape)
    global distribution
    if distribution is None:
        distribution = required_dist
    else:
        distribution = np.append(distribution, required_dist, axis = 0)

# TASK 3
def estimate_homography_best_match(kp_qry, descr_qry, kp_trg, descr_trg):
    qry_len = len(descr_qry)
    dist_descr = cdist(descr_qry, descr_trg, 'euclidean')
    print('dist_descr.dtype',dist_descr.dtype)

    min_idx = np.argmin(dist_descr, axis = 1)              # indices of the closest trg descriptors
    idx = np.arange(qry_len)
    nn_idx = np.c_[idx, min_idx]                           # nn_idx[0]: stores idx of qry descr
                                                           # nn_idx[1]: stores idx of nearest trg descr
    src_pts = np.float32([ kp_qry[m[0]] for m in nn_idx ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_trg[m[1]] for m in nn_idx ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    print('H_est',M)
    return M


# TASK 4
def make_ratio_test(kp_qry, descr_qry, kp_trg, descr_trg, H):
    qry_len = len(descr_qry)
    print('qry descr shape', descr_qry.shape)
    print('trg descr shape', descr_trg.shape)
    dist_descr = cdist(descr_qry, descr_trg, 'euclidean')
    print('dist_descr.dtype',dist_descr.dtype)

    min_idx1 = np.argmin(dist_descr, axis = 1)              # indices of the closest descriptors
    min_dist1 = dist_descr[np.arange(qry_len), min_idx1]
    kp_trg_match = kp_trg[min_idx1, :]

    # Which of these matches are correct?
    corr = get_reprojection_error(kp_qry, kp_trg_match, H)

    # Just to update what's going on:
    # We now have the indices of correct and incorrect matches as boolean values
    # so that we can we can use them to get the histogram for ratio test

    dist_descr[np.arange(qry_len), min_idx1] = float('inf')

    min_idx2 = np.argmin(dist_descr, axis = 1)              # indices of the second best match
    min_dist2 = dist_descr[np.arange(qry_len), min_idx2]

    idx1 = np.arange(qry_len)
    all = np.c_[idx1, min_idx1]
    corr_ratio = min_dist1[corr == True] / min_dist2[corr == True]
    incorr_ratio = min_dist1[corr == False] / min_dist2[corr == False]
    bins = np.arange(0,1.0001,0.05)
    corr_hist, bins = np.histogram (corr_ratio, bins)
    incorr_hist, bins = np.histogram (incorr_ratio, bins)
    print(corr_hist)

    global Gcorr, Gincorr
    Gcorr = Gcorr + corr_hist
    Gincorr = Gincorr + incorr_hist
# ...4 contd
def get_reprojection_error(kp_qry, kp_trg, H):
    kp_qry_homogenous = np.c_[kp_qry, np.ones(len(kp_qry))]
    hx = np.dot(H,kp_qry_homogenous.T)
    z = hx[-1,:]
    hx = hx/z[None,:]
    # Convert homogenous coordinates into non-homogenous
    hx = hx.T
    # print('hx.shape', hx.shape)
    kp_qry_proj = hx[:,0:2]
    # print('kp_qry_proj.shape', kp_qry_proj.shape)

    # Now we can find out the actual distances between qry and trg matches
    # print(kp_qry_proj - kp_trg_match)
    required_dist = np.linalg.norm(kp_qry_proj - kp_trg, axis=1)
    # print('repr error')
    # print(sum(required_dist<4))
    corr = (required_dist < 4)
    return corr
# ...4 contd
def ptage_false_matches_to_reject(threshold, features):
    global Gcorr, Gincorr
    poimr = None #TODO:better variable name: Percentage of incorrect matches removed
    pocmr = None # Percentage of correct matches removed
    thres = None

    if(Gincorr[19]/np.sum(Gincorr) > threshold):
        thres = 0.95
        poimr = Gincorr[19]/np.sum(Gincorr)
        pocmr = Gcorr[19]/np.sum(Gcorr)
        print('best/second_best ratio threshold is 0.95')
        print('Ptage of incorrect matches removed at 0.95: ', poimr)
        print('Ptage of correct matches removed at 0.95: ', pocmr)
    elif(np.sum(Gincorr[18:]/np.sum(Gincorr)) > threshold):
        thres = 0.90
        poimr = np.sum(Gincorr[18:])/np.sum(Gincorr)
        pocmr = np.sum(Gcorr[18:])/np.sum(Gcorr)
        print('best/second_best ratio threshold is 0.9')
        print('Ptage of incorrect matches removed at 0.9: ', poimr)
        print('Ptage of correct matches removed at 0.9: ', pocmr)
    elif(np.sum(Gincorr[17:]/np.sum(Gincorr)) > threshold):
        thres = 0.85
        poimr = np.sum(Gincorr[17:])/np.sum(Gincorr)
        pocmr = np.sum(Gcorr[17:])/np.sum(Gcorr)
        print('best/second_best ratio threshold is 0.85')
        print('Ptage of incorrect matches removed at 0.85: ', poimr)
        print('Ptage of correct matches removed at 0.85: ', pocmr)
    elif(np.sum(Gincorr[16:]/np.sum(Gincorr)) > threshold):
        thres = 0.80
        poimr = np.sum(Gincorr[16:])/np.sum(Gincorr)
        pocmr = np.sum(Gcorr[16:])/np.sum(Gcorr)
        print('best/second_best ratio threshold is 0.8')
        print('Ptage of incorrect matches removed at 0.8: ', poimr)
        print('Ptage of correct matches removed at 0.8: ', pocmr)
    elif(np.sum(Gincorr[15:]/np.sum(Gincorr)) > threshold):
        thres = 0.75
        poimr = np.sum(Gincorr[15:])/np.sum(Gincorr)
        pocmr = np.sum(Gcorr[15:])/np.sum(Gcorr)
        print('best/second_best ratio threshold is 0.75')
        print('Ptage of incorrect matches removed at 0.75: ', poimr)
        print('Ptage of correct matches removed at 0.75: ', pocmr)
    else:
        print('best/second_best ratio threshold is way too low')

    print('If hypothetically threshold is 0.95')
    poimr95 = Gincorr[19]/np.sum(Gincorr)
    pocmr95 = Gcorr[19]/np.sum(Gcorr)
    print('     Ptage of incorrect matches removed at 0.95: ', poimr95 )
    print('     Ptage of correct matches removed at 0.95: ', pocmr95)

    with open('results/experiments/' + features + '_ratio_test.txt','w') as f:
        f.write( options.features + '\n')
        f.write( 'Remove at least %s percent of the incorrect matches \n' %(str(threshold*100)))
        f.write( ' Percentage of incorrect matches removed at %s: %s \n' %(str(thres), str(poimr)) )
        f.write( ' Percentage of correct matches removed at %s: %s \n' %(str(thres), str(pocmr)) )
        f.write( '      Percentage of incr mat removed at 0.95: %s \n' %(str(poimr95)) )
        f.write( '      Percentage of corr mat removed at 0.95: %s \n' %(str(pocmr95)) )


# Task 5
def estimate_homography_ratio_test(kp_qry, descr_qry, kp_trg, descr_trg):
    qry_len = len(descr_qry)
    dist_descr = cdist(descr_qry, descr_trg, 'euclidean')

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
    # print('good', good.shape)
    if (good.shape[0] < 4):
        M = None
    else:
        src_pts = np.float32([ kp_qry[m[0]] for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_trg[m[1]] for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    print('H_est',M)
    return M















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

    options = parser.parse_args()
    print(options)

    ############## GLOBAL VARIABLE: input directory ###############
    input_dir = options.input_dir

    path_to_features1 = 'results/' + options.features + '/features'
    path_to_features2 = None
    path_to_kp = path_to_features1
    task1 = False               # Average less than 1 px distances
    task2 = True               # Histogram for NN Reprojection Error
    task3 = False               # Homography Estimation using NN
    task4 = False               # Experiments for Ratio Test
    task5 = False                # Homography Estimation using Ratio Test

    ################### Load the saved results #####################
    with open(path_to_kp, 'rb') as f:
        feature_list = pickle.load(f)
    # print('The pickle file is a list with each item storing:')
    # print('(sequence[v_boat], img_name[1.ppm], img_shape[(680, 850)], keypoints[  array([[x1,y1], [x2,y2], ...])  ])')
    no_of_sequences = int(len(feature_list)/6)

    ################# [1] Get coverage of keypoints ########################
    if task1 is True:
        print('==================== TASK 1 ==================== ')
        print('============= Spread of Less than 1 px =============')
        progress = 1
        len_kp = len(feature_list)
        for item in feature_list:
            print(progress, 'out of', len_kp, item[0], item[1])
            less_than_1_px_this = check_less_than_one(item[3])
            less_than_1_px_total = less_than_1_px_total + less_than_1_px_this
            print(' Number of KeyPoints', len(item[3]))
            # print(item[3])
            print(' Number of Distances<1', less_than_1_px_this)
            progress = progress + 1

        print('Average less than 1 px distances', less_than_1_px_total/progress)

    ################# [2] Histogram for NN Reprojection Error ########################
    if task2 is True:
        print('========================= TASK 2 ========================== ')
        print('============= Reprojrction Error of Best Match =============')
        for i in range(no_of_sequences):
            sequence = feature_list[6*i][0]
            source_dir = input_dir + '/' + sequence + '/'
            homography_name = homography_loader()
            print('Sequence In Progress: [[%d] out of %d] ...' %(i+1, no_of_sequences), sequence)

            ############ Run the test for each query target pair #############
            for outer in range(5):
                print('____[[%d]/%d]Image pair in progress: ' %(i+1, no_of_sequences), sequence, homography_name[outer])
                print(options.features)
                kp_qry = feature_list[6*i][3]
                kp_trg = feature_list[6*i+1+outer][3]
                            
                descr_qry = feature_list[6*i][4]
                descr_trg = feature_list[6*i+1+outer][4]
                
                homography_path = source_dir + '/' + homography_name[outer]
                H = np.loadtxt(homography_path)
                get_distribution_of_reprojection_error_of_matches(kp_qry, kp_trg, descr_qry, descr_trg, H)
        # If the interpixel distance is more than 15, the match is definitely wrong
        # distribution = distribution[distribution < 15] # Okay do not go by name, this is not a distribution but the actual values (like the values of a random var)
        # instead of plotting the distribution, just save the distribution as numpy file
        # plt.hist(distribution, bins = np.arange(0,15, 0.1))
        # plt.xlabel('InterPixel Distance')
        # plt.ylabel('Number of Matches')
        # plt.title('Histogram for ' + str(options.features) + ' Descriptor')
        # plt.show()
        # plt.savefig('results/' + options.features + '/best_match.png')
        distribution_file = 'results/' + options.features + '/nn_hist_' + options.features + '.npy'
        print('Saving results: ', distribution_file)
        np.save(distribution_file, distribution)

    
    ################# [3] Homography Estimation using NN ########################    
    if task3 is True:
        print('=========================== TASK 3 =========================== ')
        print('===== Homography estimation using nearest neighbours only =====')
        for i in range(no_of_sequences):
            sequence = feature_list[6*i][0]
            source_dir = input_dir + '/' + sequence + '/'
            homography_name = homography_loader()

            print('Sequence In Progress: [[%d] out of %d] ...' %(i+1, no_of_sequences), sequence)

            ############ Run the test for each query target pair #############
            for outer in range(5):
                print('____[[%d]/%d]Image pair in progress: ' %(i+1, no_of_sequences), sequence, homography_name[outer])
                print(options.features)
                kp_qry = feature_list[6*i][3]
                kp_trg = feature_list[6*i+1+outer][3]
                
                qry_img_shape = feature_list[6*i][2]
                
                descr_qry = feature_list[6*i][4]
                descr_trg = feature_list[6*i+1+outer][4]

                # print('descr.dtype', descr_qry.dtype)

                H_est = estimate_homography_best_match(kp_qry, descr_qry, kp_trg, descr_trg)

                if H_est is None:
                    h_error = None
                else:
                    homography_path = source_dir + '/' + homography_name[outer]
                    H = np.loadtxt(homography_path)
                    h_error = homography_error(H, H_est, qry_img_shape)
                    print('h_error', h_error)
                task3_h_results.append((sequence, homography_name[outer], h_error))

        # Now see how good the estimation is
        # In this case, I am going to follow the exact same methodology as SuperPoint Paper

        epsilon = [1, 2, 3, 4, 5] # Threshold for maximum permissible error in homography

        for eps in epsilon:
            v_corr = 0 # In some of the cases, homography has large errors because matches are not good enough
            v_pairs = 0
            i_corr = 0
            i_pairs = 0
            for item in task3_h_results:
                sequence = item[0]
                if sequence[0] == 'v':
                    if item[2] < eps:
                        v_corr = v_corr + 1
                    v_pairs = v_pairs + 1
                elif sequence[0] == 'i':
                    if item[2] < eps:
                        i_corr = i_corr + 1
                    i_pairs = i_pairs + 1

            total_corr = v_corr + i_corr
            total_pairs = v_pairs + i_pairs

            if ((v_pairs > 0)): 
                v_corr = v_corr/(v_pairs)

            if ((i_pairs > 0)): # and (i_pairs > i_fails)):
                i_corr = i_corr/(i_pairs)

            total_corr = total_corr/(v_pairs + i_pairs)

            result = '''
            epsilon = %s
            Ratio Test = FALSE, Use NN only
            OVERALL HOMOGRAPHY ESTIMATION RESULTS
            =====================================
            No of image pairs           %s
            Homography correct          %s
            ...
            '''
            v_result = ''' 
                Sequences with Viewpoint Changes
                -------------------------------
                No of image pairs           %s
                Homography correct          %s
            '''
            i_result = ''' 
                Sequences with Illumination Changes
                -----------------------------------
                No of image pairs           %s
                Homography correct          %s
                '''
            print('Descriptor: ', options.features)

            print( result %(str(eps), str(total_pairs), str(total_corr)) )
            print( v_result %(str(v_pairs), str(v_corr)) )
            print( i_result %(str(i_pairs), str(i_corr)) )

            with open('results/experiments/' + options.features + '_eps_' + str(eps) + '.txt','w') as f:
                f.write( options.features )
                f.write( result %(str(eps), str(total_pairs), str(total_corr)) )
                f.write( v_result %(str(v_pairs), str(v_corr)) )
                f.write( i_result %(str(i_pairs), str(i_corr)) )


    ################# [4] Experiments for Ratio Test ########################
    if task4 is True:
        print('======================== TASK 4 =============================')
        print('======== Ratio Test for Rejecting Incorrect Matches =========')

        for i in range(no_of_sequences):
            sequence = feature_list[6*i][0]
            source_dir = input_dir + '/' + sequence + '/'
            homography_name = homography_loader()
            print('Sequence In Progress: [[%d] out of %d] ...' %(i+1, no_of_sequences), sequence)

            ############ Run the test for each query target pair #############
            for outer in range(5):
                print('____[[%d]/%d]Image pair in progress: ' %(i+1, no_of_sequences), sequence, homography_name[outer])
                print(options.features)
                kp_qry = feature_list[6*i][3]
                kp_trg = feature_list[6*i+1+outer][3]
                            
                descr_qry = feature_list[6*i][4]
                descr_trg = feature_list[6*i+1+outer][4]
                
                homography_path = source_dir + '/' + homography_name[outer]
                H = np.loadtxt(homography_path)
                make_ratio_test(kp_qry, descr_qry, kp_trg, descr_trg, H)
        
        plt.plot(np.arange(0,1,0.05), Gcorr/np.sum(Gcorr))
        plt.plot(np.arange(0,1,0.05), Gincorr/np.sum(Gincorr))
        plt.xlabel('best_match/second_best_match Ratio')
        plt.ylabel('PDF')
        plt.title('Ratio test for ' + str(options.features))
        plt.savefig('results/experiments/1_rt_' + options.features + '.png')
        plt.close()
        plt.plot(np.arange(0,1,0.05), Gcorr)
        plt.plot(np.arange(0,1,0.05), Gincorr)
        plt.xlabel('best_match/second_best_match Ratio')
        plt.ylabel('Number of Matches')
        plt.title('Ratio test for ' + str(options.features))
        plt.savefig('results/experiments/2_rt_' + options.features + '.png')
        plt.close()
        # Let's find out where should we threshold to reject 90% of the false matches
        ptage_false_matches_to_reject(0.85, options.features)


    ################### [5] Homography estimation using Ratio Test #####################
    if task5 is True:
        print('============================= TASK 5 ============================ ')
        print('===== Homography estimation using ratio test - 0.7,0.8,0.95 =====')
        for i in range(no_of_sequences):
            sequence = feature_list[6*i][0]
            source_dir = input_dir + '/' + sequence + '/'
            homography_name = homography_loader()

            print('Sequence In Progress: [[%d] out of %d] ...' %(i+1, no_of_sequences), sequence)

            ############ Run the test for each query target pair #############
            for outer in range(5):
                print('____[[%d]/%d]Image pair in progress: ' %(i+1, no_of_sequences), sequence, homography_name[outer])
                print(options.features)
                kp_qry = feature_list[6*i][3]
                kp_trg = feature_list[6*i+1+outer][3]
                
                qry_img_shape = feature_list[6*i][2]
                
                descr_qry = feature_list[6*i][4]
                descr_trg = feature_list[6*i+1+outer][4]

                # print('descr.dtype', descr_qry.dtype)

                H_est = estimate_homography_ratio_test(kp_qry, descr_qry, kp_trg, descr_trg)

                if H_est is None:
                    h_error = None
                else:
                    homography_path = source_dir + '/' + homography_name[outer]
                    H = np.loadtxt(homography_path)
                    h_error = homography_error(H, H_est, qry_img_shape)
                    print('h_error', h_error)
                task5_h_results.append((sequence, homography_name[outer], h_error))

        # Now see how good the estimation is
        # In this case, I am going to follow the exact same methodology as SuperPoint Paper

        epsilon = [1, 2, 3, 4, 5] # Threshold for maximum permissible error in homography

        for eps in epsilon:
            v_corr = 0 # In some of the cases, homography has large errors because matches are not good enough
            v_pairs = 0
            i_corr = 0
            i_pairs = 0
            for item in task5_h_results:
                sequence = item[0]
                if sequence[0] == 'v':
                    v_pairs = v_pairs + 1
                    if item[2] is not None:
                        if (item[2] < eps):
                            v_corr = v_corr + 1
                elif sequence[0] == 'i':
                    i_pairs = i_pairs + 1
                    if item[2] is not None:
                        if item[2] < eps:
                            i_corr = i_corr + 1

            total_corr = v_corr + i_corr
            total_pairs = v_pairs + i_pairs

            if ((v_pairs > 0)): 
                v_corr = v_corr/(v_pairs)

            if ((i_pairs > 0)): # and (i_pairs > i_fails)):
                i_corr = i_corr/(i_pairs)

            total_corr = total_corr/(v_pairs + i_pairs)

            result = '''
            epsilon = %s
            Ratio Test = TRUE, Use best/second_best = 0.7
            OVERALL HOMOGRAPHY ESTIMATION RESULTS
            =====================================
            No of image pairs           %s
            Homography correct          %s
            ...
            '''
            v_result = ''' 
                Sequences with Viewpoint Changes
                -------------------------------
                No of image pairs           %s
                Homography correct          %s
            '''
            i_result = ''' 
                Sequences with Illumination Changes
                -----------------------------------
                No of image pairs           %s
                Homography correct          %s
                '''
            print('Descriptor: ', options.features)

            print( result %(str(eps), str(total_pairs), str(total_corr)) )
            print( v_result %(str(v_pairs), str(v_corr)) )
            print( i_result %(str(i_pairs), str(i_corr)) )

            with open('results/experiments/' + options.features + '_eps_' + str(eps) + '_rt_0.7' + '.txt','w') as f:
                f.write( options.features )
                f.write( result %(str(eps), str(total_pairs), str(total_corr)) )
                f.write( v_result %(str(v_pairs), str(v_corr)) )
                f.write( i_result %(str(i_pairs), str(i_corr)) )
