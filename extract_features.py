import argparse
import os
import time
import numpy as np
import cv2
import pickle
# selectively import other
# packages as needed inside
# functions


# Global variables
input_dir = None
results_dir = None
detector_loaded = None
cache = []



def extract_kp(detector, sequence, img_name):
    img_path = input_dir + '/' + sequence + '/' + img_name
    img = cv2.imread(img_path, 0)
    img_shape = img.shape
        
    print(detector)

    if (detector == 'sift') or (detector == 'orb'):
        kp, descr = detector_loaded.detectAndCompute(img, None)
        keypoints = cv2.KeyPoint_convert(kp)
        # print (descr)
        # print (descr.shape)

    elif detector == 'sfop':
        fillprototype(detector_loaded.mymain, ctypes.POINTER(ctypes.c_float), None)
        c_img_path = ctypes.c_char_p(img_path.encode('utf-8'))
        kp = detector_loaded.mymain(c_img_path) 	# Get the float pointer
        array_size = int(kp[0])                     # First element stores size of the entire array
        cc = np.array(np.fromiter(kp, dtype=np.float32, count=array_size))
        fillprototype(detector_loaded.free_mem, None, [ctypes.POINTER(ctypes.c_float)])
        detector_loaded.free_mem(kp)                # Free memory allocated in C
        no_of_kp = int((array_size-1)/2)            # kp = [array_size, x1, y1, x2, y2, ... ]
        keypoints = np.reshape(cc[1:array_size+1], (no_of_kp, 2))
        keypoints = np.around(keypoints, decimals = 3)
        descr = None

    elif detector == 'superpoint':
        #TODO: Replace VideoStreamer by a function that reads a single image
        vs = VideoStreamer(img_path, 0, img_shape[0], img_shape[1], 1, '*.png')
        img, status = vs.next_frame()
        pts, descr, heatmap = detector_loaded.run(img)
        keypoints = pts.T[:,0:2]
        descr = descr.T
        # print (descr)
        # print(descr.shape)        

    elif detector == 'd2net':
        # <-- D2Net Default Parameters
        import imageio
        img = imageio.imread(img_path)
        max_edge = 1600
        max_sum_edges = 2800
        preprocessing = 'caffe'
        multiscale = False
        model = detector_loaded
        # Parameters -->
        keypoints, descr = get_d2net_features(img, max_edge, max_sum_edges, preprocessing, multiscale, model)
        # print (descr)
        # print(descr.shape)        

        
    elif detector == 'lift':
        param = detector_loaded[0]
        pathconf = detector_loaded[1]
        new_kp_list = get_lift_kp(param, pathconf, img_path, img_shape)
        kps = get_lift_orientation(param, pathconf, new_kp_list, img_path)
        keypoints, descr = get_lift_features(param, pathconf, kps, img_path)

    cache.append((sequence, img_name, img_shape, keypoints, descr))
    return keypoints


def save_cache():
    print('----SAVING KEYPOINTS AND DESCRIPTORS----')
    file_path_kp = results_dir + '/features'
    with open(file_path_kp, 'wb') as f:
        pickle.dump(cache, f)


def save_img(sequence, img_name, keypoints):
    '''Function to store image with keypoints,
    this is to check whether the keypoints have
    been extracted properly. Many detectors use float 
    with dtype as dtype=float32 but the actual precision
    of floating point numbers in python is 64 bits'''

    img_path = input_dir + '/' + sequence + '/' + img_name
    img = cv2.imread(img_path)
    for i in keypoints:
        cv2.circle(img, (int(i[0]), int(i[1])), 1, (0,255,100), 2)
    cv2.imwrite(results_dir + '/' + sequence+img_name , img)

if __name__ == '__main__':
    start = time.time()
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract keypoints and feature descriptors.')

    parser.add_argument(
        '--input_dir', type=str, default='hpatches-sequences-release',
        help='Path to the input directory (must follow hpatches directory structure)'
    )
    parser.add_argument(
        '--features', type=str, default='sift',
        help='Features to extract: sift, orb, sfop, superpoint, d2net, lift'
    )
    parser.add_argument(
        '--visualize', action='store_true', default=False,
        help='Save the images with keypoints shown as dots, slows down the code (default=False)'
    )

    options = parser.parse_args()
    print(options)

    # health_check (options.input_dir, options.features, options.distance)
    # TODO: update health_check function

    ############## GLOBAL VARIABLE: input directory ###############
    input_dir = options.input_dir

    ############## GLOBAL_VARIABLE: directory to store results #####################
    results_dir = 'results/' + options.features
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ############## LOAD THE DETECTOR TO BE USED ###################

    if detector_loaded == None:
        if options.features == 'sift':
            detector_loaded = cv2.xfeatures2d.SIFT_create()

        if options.features == 'orb':
            detector_loaded = cv2.ORB_create()

        if options.features == 'sfop':
            from sfop.mysfop import fillprototype # This is needed for using ctypes pointers
            import ctypes
            detector_loaded = ctypes.cdll.LoadLibrary('sfop/build/src/libsfop.so')
            print("==> Successfully loaded shared library for SFOP")

        if options.features == 'superpoint':
            import torch
            from superpoint.demo_superpoint import SuperPointNet
            from superpoint.demo_superpoint import SuperPointFrontend
            from superpoint.demo_superpoint import VideoStreamer
            print('==> Loading [SuperPointNet] pre-trained network with default parameters: superpoint_v1.pth')
            use_cuda = torch.cuda.is_available()
            detector_loaded = SuperPointFrontend(weights_path='superpoint/superpoint_v1.pth',
                nms_dist=4,
                conf_thresh=0.015,
                nn_thresh=0.7,
                cuda=use_cuda
            )
            print('==> Successfully loaded pre-trained network.')

        if options.features == 'd2net':
            import torch
            from d2net.d2net_extract import get_d2net_features
            from d2net.lib.model_test import D2Net
            print('==> Loading [D2-net] default model: d2_tf.pth')
            cuda_available = torch.cuda.is_available()
            detector_loaded = D2Net(
                model_file='d2net/d2_tf.pth',
                use_relu=True,
                use_cuda=cuda_available
            )
            print('==> Successfully loaded pre-trained network.')

        if options.features == 'lift':
            # Using original implementation of LIFT
            # https://github.com/cvlab-epfl/LIFT
            # detector_loaded = (param, pathconfig)
            from lift.lift_detect import *
            detector_loaded = lift_model()


    ################### Load the hpatches dataset #####################
    seq_dir = next(os.walk(input_dir))[1]
    progress = 0
    for sequence in seq_dir:
        source_dir = input_dir + '/' + sequence + '/'
        print('Sequence In Progress: [[%d] out of %d] %s ...' %(progress+1,len(seq_dir), sequence))
        progress += 1

        ############ Extract keypoints and features from each image #############
        for outer in range(6):
            img_name = str(outer+1) + '.ppm'
            print('____[[%d]/%d]Image in progress: ' %(progress, len(seq_dir)), sequence, img_name)
            
            keypoints = extract_kp(options.features, sequence, img_name) # it also saves descriptors
            if options.visualize is True:
                save_img(sequence, img_name, keypoints)
            print('\n')
    
    end = time.time()
    total_time = ((end-start)/60)
    print('Total time required in min: %.4f' %(total_time))
    time_file = results_dir + '/time_to_extract_in_min.txt'


    print('Writing cached results to disk: ', results_dir)
    save_cache()
    print('Results saved!')
    print('The pickle file is a list with each item storing:')
    print('(sequence[v_boat], img_name[1.ppm], img_shape[(680, 850)], keypoints[  array([[x1,y1], [x2,y2], ...])  ], descriptors[  array([[descr1], [descr2], ...])  ])')

    # print(cache[0][0])
    # print(len(cache))
    

    with open(time_file, 'w') as f:
        f.write(str(round(total_time, 4)))