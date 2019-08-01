import argparse
import numpy as np 
import pickle
from math import isnan


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='View results for your favourite detector :)')

    parser.add_argument(
        '--features', type=str, default='sift',
        help='Detector for which you want to view the results'
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

    detector = options.features
    distance = options.distance
    threshold = options.threshold

    coverage_results_file = detector + '/kp_eval_coverage'
    wg_results_file = detector + '/kp_eval_' + distance + '_th' + str(threshold) 
    
    with open (coverage_results_file, 'rb') as f1:
        coverage_results = pickle.load(f1)

    # print(coverage_results)

    # Evaluate for all viewpoint changes:
    v_coverage = 0
    v_kp = 0
    v_total = 0
    i_coverage = 0
    i_kp = 0
    i_total = 0

    for item in coverage_results:
        sequence = item[0]
        if sequence[0] == 'v':
            v_coverage = v_coverage + item[5]
            v_kp = v_kp + item[3]
            v_total = v_total + 1
        elif sequence[0] == 'i':
            i_coverage = i_coverage + item[5]
            i_kp = i_kp + item[3]
            i_total = i_total + 1
    
    seq_total = v_total + i_total
    coverage = (v_coverage + i_coverage)/seq_total
    no_of_kp = (v_kp + i_kp)/seq_total 

    if v_total > 0:
        v_coverage = v_coverage/v_total
        v_kp = v_kp/v_total

    if i_total > 0:
        i_coverage = i_coverage/i_total
        i_kp  = i_kp/i_total

    
    result = '''
    
    OVERALL COVERAGE RESULTS
    ========================
    No of images        %s
    No of keypoints     %s
    Coverage            %s
    ...
    '''
    v_result = ''' 
        Sequences with Viewpoint Changes
        -------------------------------
        No of images        %s
        No of keypoints     %s
        Coverage            %s
    '''
    i_result = ''' 
        Sequences with Illumination Changes
        -----------------------------------
        No of images        %s
        No of keypoints     %s
        Coverage            %s

    '''
    
    print( result %(str(seq_total), str(no_of_kp), str(coverage)) )
    print( v_result %(str(v_total), str(v_kp), str(v_coverage)) )
    print( i_result %(str(i_total), str(i_kp), str(i_coverage)) )

    del coverage_results

    with open (wg_results_file, 'rb') as f2:
        wg_results = pickle.load(f2)

    # print(wg_results)

    v_UniqueMatRatio = 0
    v_QryImgSpuriousRatio = 0
    v_TrgImgSpuriousRatio = 0
    v_MultiMatRatio = 0
    v_Repeatability = 0
    v_pairs = 0
    v_QrySpu_nan_count = 0      # count the number of times nan occurs, so as to remove the same number from denominator when calculating the ratio
    v_TrgSpu_nan_count = 0

    i_UniqueMatRatio = 0
    i_QryImgSpuriousRatio = 0
    i_TrgImgSpuriousRatio = 0
    i_MultiMatRatio = 0
    i_Repeatability = 0
    i_pairs = 0
    i_QrySpu_nan_count = 0
    i_TrgSpu_nan_count = 0

    for item in wg_results:
        sequence = item[0]
        item_tuple = item[4]
        if sequence[0] == 'v':
            v_UniqueMatRatio = v_UniqueMatRatio + item_tuple[6]
            if isnan(item_tuple[7]):
                v_QrySpu_nan_count = v_QrySpu_nan_count + 1
            else:
                v_QryImgSpuriousRatio = v_QryImgSpuriousRatio + item_tuple[7]
            if isnan(item_tuple[8]):
                v_TrgSpu_nan_count = v_TrgSpu_nan_count + 1
            else:
                v_TrgImgSpuriousRatio = v_TrgImgSpuriousRatio + item_tuple[8]
            v_MultiMatRatio = v_MultiMatRatio + item_tuple[9]
            v_Repeatability = v_Repeatability + item_tuple[10]
            v_pairs = v_pairs + 1
        elif sequence[0] == 'i':
            i_UniqueMatRatio = i_UniqueMatRatio + item_tuple[6]
            if isnan(item_tuple[7]):
                i_QrySpu_nan_count = i_QrySpu_nan_count + 1
            else:
                i_QryImgSpuriousRatio = i_QryImgSpuriousRatio + item_tuple[7]
            if isnan(item_tuple[8]):
                i_TrgSpu_nan_count = i_TrgSpu_nan_count + 1
            else:
                i_TrgImgSpuriousRatio = i_TrgImgSpuriousRatio + item_tuple[8]
            i_MultiMatRatio = i_MultiMatRatio + item_tuple[9]
            i_Repeatability = i_Repeatability + item_tuple[10]
            i_pairs = i_pairs + 1

    total_pairs = v_pairs + i_pairs
    UniqueMatRatio = (v_UniqueMatRatio + i_UniqueMatRatio)/total_pairs
    QryImgSpuriousRatio = (v_QryImgSpuriousRatio + i_QryImgSpuriousRatio)/(total_pairs - v_QrySpu_nan_count - i_QrySpu_nan_count)
    TrgImgSpuriousRatio = (v_TrgImgSpuriousRatio + i_TrgImgSpuriousRatio)/(total_pairs - v_TrgSpu_nan_count - i_TrgSpu_nan_count)
    MultiMatRatio = (v_MultiMatRatio + i_MultiMatRatio)/total_pairs
    Repeatability = (v_Repeatability + i_Repeatability)/total_pairs

    if v_pairs > 0:
        v_UniqueMatRatio = v_UniqueMatRatio/v_pairs
        v_QryImgSpuriousRatio = v_QryImgSpuriousRatio/v_pairs
        v_TrgImgSpuriousRatio = v_TrgImgSpuriousRatio/v_pairs
        v_MultiMatRatio = v_MultiMatRatio/v_pairs
        v_Repeatability = v_Repeatability/v_pairs

    if i_pairs >0:
        i_UniqueMatRatio = i_UniqueMatRatio/i_pairs
        i_QryImgSpuriousRatio = i_QryImgSpuriousRatio/i_pairs
        i_TrgImgSpuriousRatio = i_TrgImgSpuriousRatio/i_pairs
        i_MultiMatRatio = i_MultiMatRatio/i_pairs
        i_Repeatability = i_Repeatability/i_pairs

    result = '''
    
    OVERALL KEYPOINT MATCHING RESULTS
    =================================
    Image Pairs                 %s
    Unique Match Ratio          %s
    Spurious KP Ratio Qry       %s
    Spurious KP Ratio Trg       %s
    Multiple Match Ratio        %s
    Repeatability               %s
    ...
    '''
    v_result = ''' 
        Sequences with Viewpoint Changes
        -------------------------------
        Image Pairs                 %s
        Unique Match Ratio          %s
        Spurious KP Ratio Qry       %s
        Spurious KP Ratio Trg       %s
        Multiple Match Ratio        %s
        Repeatability               %s
    '''
    i_result = ''' 
        Sequences with Illumination Changes
        -----------------------------------
        Image Pairs                 %s
        Unique Match Ratio          %s
        Spurious KP Ratio Qry       %s
        Spurious KP Ratio Trg       %s
        Multiple Match Ratio        %s
        Repeatability               %s

    '''
    print('''==============================================================================''')
    print( result %(str(total_pairs), str(UniqueMatRatio), str(QryImgSpuriousRatio), str(TrgImgSpuriousRatio), str(MultiMatRatio), str(Repeatability)) )
    print( v_result %(str(v_pairs), str(v_UniqueMatRatio), str(v_QryImgSpuriousRatio), str(v_TrgImgSpuriousRatio), str(v_MultiMatRatio), str(v_Repeatability)) )
    print( i_result %(str(i_pairs), str(i_UniqueMatRatio), str(i_QryImgSpuriousRatio), str(i_TrgImgSpuriousRatio), str(i_MultiMatRatio), str(i_Repeatability)) )