import argparse
import numpy as np 
import pickle


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='View results for your favourite descriptor :)')

    parser.add_argument(
        '--features', type=str, default='sift',
        help='Detector for which you want to view the results'
    )

    options = parser.parse_args()
    print(options)

    options = parser.parse_args()
    print(options)

    descriptor = options.features
    descr_results_file = descriptor + '/descr_eval_homography'

    with open (descr_results_file, 'rb') as f:
        homography_results = pickle.load(f)

    v_error = 0
    v_fails = 0 # In some of the cases, homography cannot be estimated because matches are not good enough
    v_pairs = 0
    i_error = 0
    i_fails = 0
    i_pairs = 0

    for item in homography_results:
        sequence = item[0]
        if sequence[0] == 'v':
            if item[2] == None:
                v_fails = v_fails + 1
            else:
                v_error = v_error + item[2]
            v_pairs = v_pairs + 1
        elif sequence[0] == 'i':
            if item[2] == None:
                i_fails = i_fails + 1
            else:
                i_error = i_error + item[2]
            i_pairs = i_pairs + 1

    total_fails = v_fails + i_fails
    total_pairs = v_pairs + i_pairs
    total_error = (v_error + i_error)/(total_pairs - total_fails)

    if ((v_pairs > 0) and (v_pairs > v_fails)):
        v_error = v_error/(v_pairs - v_fails)

    if ((i_pairs > 0) and (i_pairs > i_fails)):
        i_error = i_error/(i_pairs - i_fails)

    result = '''
    
    OVERALL HOMOGRAPHY ESTIMATION RESULTS
    =====================================
    No of image pairs           %s
    Out of which failures       %s
    Average homography error    %s
    ...
    '''
    v_result = ''' 
        Sequences with Viewpoint Changes
        -------------------------------
        No of image pairs           %s
        Out of which failures       %s
        Average homography error    %s
    '''
    i_result = ''' 
        Sequences with Illumination Changes
        -----------------------------------
        No of image pairs           %s
        Out of which failures       %s
        Average homography error    %s

    '''
    print('Descriptor: ', descriptor)

    print( result %(str(total_pairs), str(total_fails), str(total_error)) )
    print( v_result %(str(v_pairs), str(v_fails), str(v_error)) )
    print( i_result %(str(i_pairs), str(i_fails), str(i_error)) )