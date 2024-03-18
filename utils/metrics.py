"""
author : Peb Ruswono Aryan
metric for evaluating binarization algorithms
implemented : 
 * F-Measure
 * pseudo F-Measure (as in H-DIBCO 2010 & 2012)
 * Peak Signal to Noise Ratio (PSNR)
 * Negative Rate Measure (NRM)
 * Misclassification Penaltiy Measure (MPM)
 * Distance Reciprocal Distortion (DRD)
usage:
	python metric.py test-image.png ground-truth-image.png
"""
import numpy as np 
import cv2
# uses https://gist.github.com/pebbie/c2cec958c248339c8537e0b4b90322da for skeletonization

import os.path as path
import sys

import numpy as np
from scipy import ndimage as ndi

# lookup tables for bwmorph_thin

G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=bool)

def bwmorph_thin(image, n_iter=None):
    """
    Perform morphological thinning of a binary image
    
    Parameters
    ----------
    image : binary (M, N) ndarray
        The image to be thinned.
    
    n_iter : int, number of iterations, optional
        Regardless of the value of this parameter, the thinned image
        is returned immediately if an iteration produces no change.
        If this parameter is specified it thus sets an upper bound on
        the number of iterations performed.
    
    Returns
    -------
    out : ndarray of bools
        Thinned image.
    
    See also
    --------
    skeletonize
    
    Notes
    -----
    This algorithm [1]_ works by making multiple passes over the image,
    removing pixels matching a set of criteria designed to thin
    connected regions while preserving eight-connected components and
    2 x 2 squares [2]_. In each of the two sub-iterations the algorithm
    correlates the intermediate skeleton image with a neighborhood mask,
    then looks up each neighborhood in a lookup table indicating whether
    the central pixel should be deleted in that sub-iteration.
    
    References
    ----------
    .. [1] Z. Guo and R. W. Hall, "Parallel thinning with
           two-subiteration algorithms," Comm. ACM, vol. 32, no. 3,
           pp. 359-373, 1989.
    .. [2] Lam, L., Seong-Whan Lee, and Ching Y. Suen, "Thinning
           Methodologies-A Comprehensive Survey," IEEE Transactions on
           Pattern Analysis and Machine Intelligence, Vol 14, No. 9,
           September 1992, p. 879
    
    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square[0,1] =  1
    >>> square
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> skel = bwmorph_thin(square)
    >>> skel.astype(np.uint8)
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    # check that we have a 2d binary image, and convert it
    # to uint8
    skel = np.array(image).astype(np.uint8)
    
    if skel.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    # neighborhood mask
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(skel) # count points before thinning
        
        # for each subiteration
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            skel[D] = 0
            
        after = np.sum(skel) # coint points after thinning
        
        if before == after:
            # iteration had no effect: finish
            break
            
        # count down to iteration limit (or endlessly negative)
        n -= 1
    
    return skel.astype(bool)
    
def drd_fn(im, im_gt):
	height, width = im.shape
	neg = np.zeros(im.shape)
	neg[im_gt!=im] = 1
	y, x = np.unravel_index(np.flatnonzero(neg), im.shape)
	
	n = 2
	m = n*2+1
	W = np.zeros((m,m), dtype=np.uint8)
	W[n,n] = 1.
	W = cv2.distanceTransform(1-W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	W[n,n] = 1.
	W = 1./W
	W[n,n] = 0.
	W /= W.sum()
	
	nubn = 0.
	block_size = 8
	for y1 in range(0, height, block_size):
		for x1 in range(0, width, block_size):
			y2 = min(y1+block_size-1,height-1)
			x2 = min(x1+block_size-1,width-1)
			block_dim = (x2-x1+1)*(y1-y1+1)
			block = 1-im_gt[y1:y2, x1:x2]
			block_sum = np.sum(block)
			if block_sum>0 and block_sum<block_dim:
				nubn += 1

	drd_sum= 0.
	tmp = np.zeros(W.shape)
	for i in range(min(1,len(y))):
		tmp[:,:] = 0 

		x1 = max(0, x[i]-n)
		y1 = max(0, y[i]-n)
		x2 = min(width-1, x[i]+n)
		y2 = min(height-1, y[i]+n)

		yy1 = y1-y[i]+n
		yy2 = y2-y[i]+n
		xx1 = x1-x[i]+n
		xx2 = x2-x[i]+n

		tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(im[y[i],x[i]]-im_gt[y1:y2+1,x1:x2+1])
		tmp *= W

		drd_sum += np.sum(tmp)
	return drd_sum/nubn

from torchvision.transforms.functional import rgb_to_grayscale

def compute_metrics_DIBCO(im, im_gt):
    #im =  rgb_to_grayscale(im)# im_gt.convert('L')
    #im_gt = rgb_to_grayscale(im_gt)#im_gt.convert('L')
    im = im[0].numpy()
    th, im = cv2.threshold(im, 0.5, 1, cv2.THRESH_BINARY)
    im_gt = im_gt[0].numpy()    
    th, im_gt = cv2.threshold(im_gt, 0.5, 1, cv2.THRESH_BINARY)
    height, width = im.shape
    npixel = height*width
    
    im[im>0] = 1
    im_gt[im_gt>0] = 1
    sk = bwmorph_thin(1-im_gt)
    im_sk = np.ones(im_gt.shape)
    im_sk[sk] = 0
	


    ptp = np.zeros(im_gt.shape)
    ptp[(im==0) & (im_sk==0)] = 1
    numptp = ptp.sum()

    tp = np.zeros(im_gt.shape)
    tp[(im==0) & (im_gt==0)] = 1
    numtp = tp.sum()
    if numtp==0:    
        numtp=1

    tn = np.zeros(im_gt.shape)
    tn[(im==1) & (im_gt==1)] = 1
    numtn = tn.sum()    
    fp = np.zeros(im_gt.shape)
    fp[(im==0) & (im_gt==1)] = 1
    numfp = fp.sum()    
    fn = np.zeros(im_gt.shape)
    fn[(im==1) & (im_gt==0)] = 1
    numfn = fn.sum()    
    precision = numtp / (numtp + numfp)
    recall = numtp / (numtp + numfn)
    precall = numptp / np.sum(1-im_sk)
    fmeasure = (2*recall*precision)/(recall+precision)

    pfmeasure = (2*precall*precision)/(precall+precision)   

    mse = (numfp+numfn)/npixel
    psnr = 10.*np.log10(1./mse) 
    psnr2 = psnr_2(im, im_gt)
    drd = drd_fn(im, im_gt) 
    print(f"psnr2 = {psnr2}")
    print("F-measure\t: {0}\npF-measure\t: {1}\nPSNR\t\t: {2}\nDRD\t\t: {3}".format(fmeasure, pfmeasure, psnr, drd))
    #my_compute_precision_recall_fmeasure(im, im_gt,im_sk)
    return fmeasure, pfmeasure, psnr, drd

import math
def psnr_2(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

if __name__ == '__main__':
    import os
    folder_path="/mnt/media/giordano/DBICO_Results"
    gt_folder_path = "/mnt/media/giordano/DIBCODATASET2019/test_gt"
    
    for image in os.listdir(gt_folder_path):
        im = cv2.imread(os.path.join(folder_path,image+".png"),0)
        im_gt = cv2.imread(os.path.join(gt_folder_path,image),0)
        #print(im.shape)
        #print(im_gt.shape)
        #print(os.path.join(gt_folder_path,image))
        compute_metrics_DIBCO(im, im_gt)
        
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import  remove_small_holes, remove_small_objects

def my_compute_precision_recall_fmeasure(u, u0_GT, u0_SKL_GT=None):
    xm, ym = u.shape

    if u0_GT.size == 0:
        u0_GT = np.full((xm, ym), np.nan)

    if u0_SKL_GT is None:
        u0_SKL_GT = remove_thin(remove_small_holes(remove_small_objects(~u0_GT)))

    # TP pixels
    temp_tp = np.logical_and(u == 0, u0_GT == 0)
    # FP pixels
    temp_fp = np.logical_and(u == 0, u0_GT != 0)
    # FN pixels
    temp_fn = np.logical_and(u != 0, u0_GT == 0)
    # TN pixels
    temp_tn = np.logical_and(u != 0, u0_GT != 0)
    # SKL TP / FN pixels
    temp_skl_tp = np.logical_and(u == 0, u0_SKL_GT == 0)
    temp_skl_fp = np.logical_and(u == 0, u0_SKL_GT != 0)
    temp_skl_fn = np.logical_and(u != 0, u0_SKL_GT == 0)
    temp_skl_tn = np.logical_and(u != 0, u0_SKL_GT != 0)

    # counts
    count_tp = np.sum(temp_tp)
    count_fp = np.sum(temp_fp)
    count_fn = np.sum(temp_fn)
    count_tn = np.sum(temp_tn)
    count_skl_tp = np.sum(temp_skl_tp)
    count_skl_fp = np.sum(temp_skl_fp)
    count_skl_fn = np.sum(temp_skl_fn)
    count_skl_tn = np.sum(temp_skl_tn)

    # precision
    temp_p = count_tp / (count_fp + count_tp)
    # recall
    temp_r = count_tp / (count_fn + count_tp) if count_tp != 0 else np.nan
    # p-recall
    temp_pseudo_p = count_skl_tp / (count_skl_fp + count_skl_tp)
    temp_pseudo_r = count_skl_tp / (count_skl_fn + count_skl_tp) if count_skl_tp != 0 else np.nan
    # f-measure
    temp_f = 100 * 2 * (temp_p * temp_r) / (temp_p + temp_r) if (temp_p + temp_r) != 0 else 0
    # p-f-measure
    temp_pseudo_f = 100 * 2 * (temp_p * temp_pseudo_r) / (temp_p + temp_pseudo_r) if (temp_p + temp_pseudo_r) != 0 else 0
    # sensetivity
    temp_sens = count_tp / (count_tp + count_fn) if (count_tp + count_fn) != 0 else np.nan
    # specificity
    temp_spec = count_tn / (count_tn + count_fp) if (count_tn + count_fp) != 0 else np.nan
    # BCR: Balanced Classification Rate
    temp_BCR = 0.5 * (temp_sens + temp_spec) if not (np.isnan(temp_sens) or np.isnan(temp_spec)) else np.nan
    # AUC: Area Under the Curve
    temp_AUC = 0.5 * (temp_sens + temp_spec) if not (np.isnan(temp_sens) or np.isnan(temp_spec)) else np.nan
    # BER: Balanced Error Rate
    temp_BER = 100 * (1 - temp_BCR) if not np.isnan(temp_BCR) else np.nan
    # S-F-measure: harmonic mean of sensetivity and specificity
    temp_s_f_measure = 100 * 2 * (temp_sens * temp_spec) / (temp_sens + temp_spec) if (temp_sens + temp_spec) != 0 else 0
    # Accuracy: mean of sensetivity and specificity
    temp_accu = (count_tp + count_tn) / (count_tp + count_tn + count_fp + count_fn) if (count_tp + count_tn + count_fp + count_fn) != 0 else np.nan
    # gAccuracy: Geometric mean of sensetivity and specificity
    temp_g_accu = np.sqrt(temp_sens * temp_spec) if not (np.isnan(temp_sens) or np.isnan(temp_spec)) else np.nan
    # NRM (Negative Rate Metric) (*10^-2)
    NR_FN = count_fn / (count_fn + count_tp) if (count_fn + count_tp) != 0 else np.nan
    NR_FP = count_fp / (count_fp + count_tn) if (count_fp + count_tn) != 0 else np.nan
    temp_NRM = (NR_FN + NR_FP) / 2 if not (np.isnan(NR_FN) or np.isnan(NR_FP)) else np.nan

    # PSNR
    err = np.sum(temp_fp | temp_fn) / (xm * ym) if (xm * ym) != 0 else 0
    temp_PSNR = 10 * np.log10(1 / err) if err != 0 else np.nan

    # DRD: Distance Reciprocal Distortion Metric
    blkSize = 8  # even number
    MaskSize = 5  # odd number
    u0_GT1 = np.zeros((xm + 2, ym + 2), dtype=bool)
    u0_GT1[1:xm + 1, 1:ym + 1] = u0_GT
    intim = np.cumsum(np.cumsum(u0_GT1, axis=0), axis=1)
    NUBN = 0
    blkSizeSQR = blkSize ** 2
    for i in range(1, xm, xm - blkSize + 1):
        for j in range(1, ym, ym - blkSize + 1):
            blkSum = intim[i + blkSize - 1, j + blkSize - 1] - intim[i - 1, j + blkSize - 1] - intim[i + blkSize - 1, j - 1] + intim[i - 1, j - 1]
            if blkSum == 0 or blkSum == blkSizeSQR:
                pass
            else:
                NUBN += 1

    wm = np.zeros((MaskSize, MaskSize))
    ic = (MaskSize + 1) // 2
    jc = ic  # center coordinate
    for i in range(1, MaskSize + 1):
        for j in range(1, MaskSize + 1):
            wm[i - 1, j - 1] = 1 / np.sqrt((i - ic) ** 2 + (j - jc) ** 2)
    wm[ic - 1, jc - 1] = 0
    wnm = wm / np.sum(wm)

    u0_GT_Resized = np.zeros((xm + ic, ym + jc))
    u0_GT_Resized[ic - 1:xm + ic - 1, jc - 1:ym + jc - 1] = u0_GT
    u_Resized = np.zeros((xm + ic, ym + jc))
    u_Resized[ic - 1:xm + ic - 1, jc - 1:ym + jc - 1] = u
    temp_fp_Resized = np.logical_and(u_Resized == 0, u0_GT_Resized != 0)
    temp_fn_Resized = np.logical_and(u_Resized != 0, u0_GT_Resized == 0)
    Diff = temp_fp_Resized | temp_fn_Resized
    xm2, ym2 = Diff.shape
    SumDRDk = 0
    for i in range(ic - 1, xm2 - ic + 1):
        for j in range(jc - 1, ym2 - jc + 1):
            if Diff[i, j] == 1:
                Local_Diff = my_xor_infile(u0_GT_Resized[i - ic + 1:i + ic, j - ic + 1:j + ic], u_Resized[i, j])
                DRDk = np.sum(np.sum(Local_Diff * wnm))
                SumDRDk += DRDk

    temp_DRD = SumDRDk / NUBN if NUBN != 0 else np.nan


    # output
    temp_obj_eval = {
        'Precision': temp_p,
        'Recall': temp_r,
        'Fmeasure': temp_f,
        'P_Precision': temp_pseudo_p,
        'P_Recall': temp_pseudo_r,
        'P_Fmeasure': temp_pseudo_f,
        'Sensitivity': temp_sens,
        'Specificity': temp_spec,
        'BCR': temp_BCR,
        'AUC': temp_AUC,
        'BER': temp_BER,
        'SFmeasure': temp_s_f_measure,
        'Accuracy': temp_accu,
        'GAccuracy': temp_g_accu,
        'NRM': temp_NRM,
        'PSNR': temp_PSNR,
        'DRD': temp_DRD,

    }
    print(temp_obj_eval)
    return temp_obj_eval


def my_xor_infile(u_infile, u0_GT_infile):
    # Reza
    temp_fp_infile = np.logical_and(u_infile == 0, u0_GT_infile != 0)
    temp_fn_infile = np.logical_and(u_infile != 0, u0_GT_infile == 0)
    temp_xor_infile = temp_fp_infile | temp_fn_infile

    return temp_xor_infile
        
        
        
        