# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
from config import cfg



from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    #print('preds:' , preds)
    #print('targets: ',target)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    dists_ = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]): #128 - batch
        for c in range(preds.shape[1]): # 17 - kpt class
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :]
                normed_targets = target[n, c, :]
                #print('normed_preds: ', normed_preds)
                #print('normed_targets', normed_targets)
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                
            else:
                dists[c, n] = -1
    #print('dists shape:' , dists.shape) --> (17, 128)
    #exit(0)
    #print(dists)
    return dists

# normalize head size
# select the value (PCKh??)
# make "dists" value to ratios depending on head size

def dist_acc(dists, thr): # dist=(128)
    ''' Return percentage below threshold while ignoring values with a -1 '''   
    num_dist_cal = 0
    correct_pred = 0
    for i in range(len(dists)):
    	if dists[i] != -1:
  	    num_dist_cal += 1
  	    if(dists[i] < thr[i]):
  	    	correct_pred += 1
    
    if num_dist_cal > 0 :
    	return correct_pred / num_dist_cal
    else:
    	return -1  	    
  	    
  	    
def accuracy(output, target, head_size,width, height, hm_type='gaussian', thr=0.5):

    image_size = np.array(cfg.MODEL.IMAGE_SIZE)
    heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
    '''
    Calculate accuracy according to PCK, [PCK-->PCK-h]
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    #print('output.shape: ', output.shape) #--> (64,48)

    idx = list(range(output.shape[1]))
    norm = 1.0
    #print('first_pred: ', output)
    
 
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output) # 128, 17, 2
        #print('pred :',type(pred)) 
        #exit()
    
        
        target, _ = get_max_preds(target) # 128, 17, 2 
        #### EXAMPLE
        #pred = [10, 10]
        #head_size = [10]
        #target = [15, 10]
        
        #print('target :', target.shape) 
        #exit(0)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10 # 6.4, 4.8
        #print('output:', output.shape) #--> (128, 17, 64, 48)
        #exit(0)
        
        #normalize head_size
        #print('head_size: ', head_size)
        #feat_stride = image_size / heatmap_size
        #head_size[0] = head_size[0] / feat_stride[0]
        #head_size[1] = head_size[1] / feat_stride[1]
        head_h = head_size[0] / (height / h)
        head_w = head_size[1] / (width / w)
        #print('norm_head_h_size: ', head_h)
        #print('norm_head_w_size: ', head_w)
        
        threshold = np.zeros(len(head_size[0]))

        for i in range(len(head_size[0])):
        	threshold[i] = (math.sqrt(math.pow(head_h[i],2) + math.pow(head_w[i],2)) * 0.6) * thr        
        #print('threshold: ', threshold)
        
    dists = calc_dists(pred, target, norm) # (17, 128)

    acc = np.zeros((len(idx) + 1)) #acc 18
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)): # by class
        #acc[i + 1] = dist_acc(dists[idx[i]])
        acc[i + 1] = dist_acc(dists[idx[i]],threshold)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0

    if cnt != 0:
        acc[0] = avg_acc
    #print('acc: ' , acc)    
    
    
    return acc, avg_acc, cnt, pred

