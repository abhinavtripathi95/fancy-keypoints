import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats.mstats import hmean

def health_check(input_dir, detector, distance):
    available_distances = ['euclidean',
                        'mahalanobis',
                        'cityblock']
    available_detectors = ['sift', 'orb', 'sfop',
                        'superpoint', 'd2net', 'lift']
    
    if not (distance in available_distances):
        raise Exception('{} distance not supported', format(distance))

    if not (detector in available_detectors):
        raise Exception('{} detector not supported', format(detector))

    if not (os.path.exists(input_dir)):
        raise Exception('Input directory not found {}', format(input_dir))

    print('==> Health check successfully completed.')


def homography_loader():
    homography_path = []
    homography_path.append('H_1_2')
    homography_path.append('H_1_3')
    homography_path.append('H_1_4')
    homography_path.append('H_1_5')
    homography_path.append('H_1_6')
    return homography_path


def get_overlapping_kp(point_qry, point_trg, H, qry_img_shape, trg_img_shape):
    Hinv = np.linalg.inv(H)
    ############ Overlay the images to find the ##############
    ######### keypoints in the overlapping region ############
    homogenous_qry = np.transpose(np.c_[point_qry, np.ones(len(point_qry))])
    hx = np.dot(H, homogenous_qry)
    print('hx.dtype',hx.dtype)
    z = hx[-1,:]
    hx = hx/z[None,:]
    # Convert homogenous coordinates into non-homogenous
    hx = hx[0:2,:]
    # print(point_qry.shape)
    # print(hx.shape)
    point_qry = point_qry[(hx[0] >= 0) & (hx[0] <= trg_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= trg_img_shape[0]), :]
    point_qry_proj_on_trg = hx[:, (hx[0] >= 0) & (hx[0] <= trg_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= trg_img_shape[0])] 
    point_qry_proj_on_trg = np.transpose(point_qry_proj_on_trg)
    
    homogenous_trg = np.transpose(np.c_[point_trg, np.ones(len(point_trg))])
    hx = np.dot(Hinv, homogenous_trg)            
    z = hx[-1,:]
    hx = hx/z[None,:]
    # Convert homogenous coordinates into non-homogenous
    hx = hx[0:2,:]
    point_trg = point_trg[(hx[0] >= 0) & (hx[0] <= qry_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= qry_img_shape[0]), :]
    point_trg_proj_on_qry = hx[:, (hx[0] >= 0) & (hx[0] <= qry_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= qry_img_shape[0])]
    point_trg_proj_on_qry = np.transpose(point_trg_proj_on_qry)
    point_qry_len = len(point_qry)
    point_trg_len = len(point_trg)
    print('    Overlapping kp in query image: ', point_qry_len)
    print('    Overlapping kp in target image: ', point_trg_len)
    return point_qry, point_trg, point_qry_proj_on_trg, point_trg_proj_on_qry
    # NOTE: point_qry_proj_on_trg stores the keypoints overlayed on the 
    # target image, however point_qry stores the coordinates on the original
    # query image



def get_dist(point_qry, point_trg, dist_metric):
    dist = cdist(point_qry, point_trg, dist_metric)
    return dist

def evaluate_matches(dist_in_trg_img, dist_in_qry_img, point_qry_len, point_trg_len, threshold):
    dist = dist_in_trg_img
    if point_qry_len == 0:
        UniqueMat = 0
        SpuriousQry = 0
        SpuriousTrg = point_trg_len
        MultiMat = 0
        UniqueMatRatio = 0
        QryImgSpuriousRatio = float('nan')
        TrgImgSpuriousRatio = 1.0
        MultiMatRatio = 0
        Repeatability = 0

    elif point_trg_len == 0:
        UniqueMat = 0
        SpuriousQry = point_qry_len
        SpuriousTrg = 0
        MultiMat = 0
        UniqueMatRatio = 0
        QryImgSpuriousRatio = 1.0
        TrgImgSpuriousRatio = float('nan')
        MultiMatRatio = 0
        Repeatability = 0

    else:
        adj_mat = np.zeros(dist.shape)
        adj_mat[(dist>=0) & (dist<=threshold)] = 1

        adj_mat_on_qry = np.zeros(dist_in_qry_img.shape)
        adj_mat_on_qry[(dist_in_qry_img >= 0) & (dist_in_qry_img <= threshold)] = 1
        # dist[dist>1] = float('nan')
        # if histogram:
        #     get_histogram(dist, results_dir, threshold)

        sum_i = np.sum(adj_mat, axis = 1) # i = 0 to qry kp-1
        SpuriousQry = np.count_nonzero(sum_i==0)
        sum_j = np.sum(adj_mat, axis = 0) # j = 0 to trg jp-1
        SpuriousTrg = np.count_nonzero(sum_j==0)

        rowmask = sum_i==1
        colmask = sum_j==1
        UniqueMat = np.sum((adj_mat * rowmask[:,None]) * colmask[:,None].T)
        MultiMat = np.sum(adj_mat) - UniqueMat

        QryImgSpuriousRatio = SpuriousQry/point_qry_len
        TrgImgSpuriousRatio = SpuriousTrg/point_trg_len

        UniqueMatRatio = UniqueMat/(min(point_qry_len, point_trg_len))
        MultiMatRatio = MultiMat/(point_qry_len + point_trg_len)

        '''
        Adding repeatability as a criteria here
        Correctness wrt a keypoint => at least one 1 in the adjacency
        matrix
        '''
        corr_i = np.sum(sum_i>0)
        sum_i_on_qry = np.sum(adj_mat_on_qry, axis = 0)
        corr_i_on_qry = np.sum(sum_i_on_qry>0)
        Repeatability = (corr_i + corr_i_on_qry)/(point_qry_len + point_trg_len)
    
    
    print('    Unique Matches: ', UniqueMat)
    print('    Spurious Keypoints in Query Img: ', SpuriousQry)
    print('    Spurious Keypoints in Target Img: ', SpuriousTrg)
    print('    Multiple Matches: ', MultiMat) 
    print('    Unique Match Ratio: ', UniqueMatRatio)
    print('    QryImg Spurious Ratio: ', QryImgSpuriousRatio)
    print('    TrgImg Spurious Ratio: ', TrgImgSpuriousRatio)
    print('    Multiple Match Ratio: ', MultiMatRatio)
    print('    Repeatability: ', Repeatability)

    eval_results = (point_qry_len, point_trg_len, UniqueMat, SpuriousQry, SpuriousTrg, MultiMat, UniqueMatRatio, QryImgSpuriousRatio, TrgImgSpuriousRatio, MultiMatRatio, Repeatability)
    return eval_results



def get_coverage(item):
    '''A function to get coverage metric
    from the stored list of keypoints
    '''
    sequence = item[0]
    img_name = item[1]
    img_shape = item[2]
    kp = item[3]
    # print(kp)
    no_of_kp = len(kp)
    distances = cdist(kp, kp, 'euclidean')

    hm_i = []
    for i in range(no_of_kp):
        dist_i = distances[i,:]
        # You don't want to take all the distances into account, so just take the ones which are
        # at least 1 pixel away
        hm_i.append ( hmean(dist_i[dist_i > 1]) ) 
    hm = hmean(hm_i)
    hm_normalized = hm/np.sqrt(img_shape[0] * img_shape[1])
    print('Coverage: ', hm_normalized)
    return sequence, img_name, img_shape, no_of_kp, hm, hm_normalized

