import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import cv2
import mmdet
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor, build_dataset, build_dataloader
from mmdet.datasets.pipelines import Compose
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_xyxy_to_cxcywh
from tqdm import tqdm
import pickle
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
import lap
import json
import argparse
import os
import csv
from torch.distributions.normal import Normal



parser = argparse.ArgumentParser()

parser.add_argument("--gt_pkl_filename", help="Pickle file containing ground truth.",
                    type=str)
parser.add_argument('--result_output_filename', help="Output CSV filename", 
                    type=str)
parser.add_argument('--score_threshold', help="Score threshold used in merging.", 
                    type=float, default=0.0)
parser.add_argument('--iou_threshold', help='Minimum IoU threshold for matching', 
                    type=float, default=0.5)
parser.add_argument('--result_identifiers', type=str, help='Additional identifiers for CSV', default=None)


args = parser.parse_args()

def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    matches = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(matches[:, 1])
    matches = matches[sort_idx]
    matches = torch.from_numpy(matches).long()
    return matches

def get_iou_coverage_sharpness(old_bboxes, new_bboxes):
        coverage_array = []
        sharpness_array = []
        iou_array = []

        old_bboxes = np.asarray(old_bboxes)
        new_bboxes = np.asarray(new_bboxes)
       
        for idx in range(len(new_bboxes)):
            import pdb

            nx1 = new_bboxes[idx, -4] 
            ny1 = new_bboxes[idx, -3] 
            nx2 = new_bboxes[idx, -2] 
            ny2 = new_bboxes[idx, -1] 
           
            #find the iou with the detection bbox
            ox1 = old_bboxes[idx, 0] 
            oy1 = old_bboxes[idx, 1]
            ox2 = old_bboxes[idx, 2]
            oy2 = old_bboxes[idx, 3]

            #centroids, width and heights
            ncx = (nx1+nx2)/2
            ncy = (ny1+ny2)/2
            nw = nx2-nx1
            nh = ny2-ny1

            ocx = (ox1+ox2)/2
            ocy = (oy1+oy2)/2
            ow = ox2-ox1
            oh = oy2-oy1
            
            ### 1 is good, 0 is bad
            xx1 = max(nx1, ox1)
            yy1 = max(ny1, oy1)
            xx2 = min(nx2, ox2)
            yy2 = min(ny2, oy2)

            w = xx2 - xx1
            h = yy2 - yy1

            inter = w*h
            Narea = (nx2-nx1)*(ny2-ny1)
            Oarea = (ox2-ox1)*(oy2-oy1)
            union = Narea + Oarea - inter
            IoU = inter/union
            coverage = inter/Oarea #Similar to recall
            sharpness = inter/Narea #Similar to precision
            iou_array.append(IoU)
            coverage_array.append(coverage)
            sharpness_array.append(sharpness)
            
        return np.array(iou_array), np.array(coverage_array), np.array(sharpness_array)

    
with open(args.gt_pkl_filename, 'rb') as f:
    model_output = pickle.load(f)
    

fnames = [o['ori_filename'] for o in model_output]


all_ious, all_coverage, all_sharpness = [], [], []
matched_ious, matched_coverage, matched_sharpness = [], [], []
count_unmatched = 0
count_no_preds = 0
total_gt_bboxes = 0
unmatched_gt_boxes = []

IOU_THRESHOLD = args.iou_threshold

for sample in model_output:
    fname = sample['ori_filename']
    H, W, _ = sample['ori_shape']
    gt_bboxes = torch.from_numpy(sample['gt_bboxes'])
    gt_labels = torch.from_numpy(sample['gt_labels']).long()    
    probs = torch.from_numpy(sample['cls_probs'])[-1]    

    is_bg = (torch.argmax(probs, dim=-1) == (probs.shape[1] - 1)) 
    max_probs, _ = probs.max(-1)
    is_conf = max_probs >= args.score_threshold

    mask = ~is_bg & is_conf
    probs = probs[mask]

    H, W, _ = sample['ori_shape']
    factor = torch.tensor([W, H, W, H]).unsqueeze(0)
    bbox_preds = torch.from_numpy(sample['bbox_preds'][-1]) 
    bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds) * factor
    
    bbox_preds = bbox_preds[mask]
    
    if len(bbox_preds) == 0:
        count_no_preds += 1
        all_ious.append(np.zeros(len(gt_bboxes)))
        all_coverage.append(np.zeros(len(gt_bboxes)))
        all_sharpness.append(np.zeros(len(gt_bboxes)))
        continue
    
    #compute pairwise iou between all predictions and gt
    #matrix has shape N_preds x N_gt (for detr, N_preds == 100)
    ious = BboxOverlaps2D()(bbox_preds, gt_bboxes)

    #set iou to 0 for all boxes predicted as background (optional?)
    #bg_mask = probs[:, -1] >= 0.5 
    #ious[bg_mask] = 0
    
    #run min cost assignment algo using negative iou 
    #matrix has shape N_gt x 2
    original_matches = matches = linear_assignment(1.-ious.numpy())
    sel_indices = ious.numpy()[matches.numpy()[:, 0], matches.numpy()[:, 1]] >= IOU_THRESHOLD
    matches = matches[sel_indices]
    
    #all gt boxes should be matched to something (could be background)
    if len(matches) != len(gt_bboxes):

        count_unmatched += len(gt_bboxes) - len(matches)
        iou_array, coverage_array, sharpness_array = get_iou_coverage_sharpness(old_bboxes=gt_bboxes[matches[:, 1]], new_bboxes=bbox_preds[matches[:, 0]])
        all_ious.append(iou_array)
        all_coverage.append(coverage_array)
        all_sharpness.append(sharpness_array)
        matched_ious.append(iou_array)
        matched_coverage.append(coverage_array)
        matched_sharpness.append(sharpness_array)
        unmatched_gt_indxs = np.setdiff1d(np.arange(len(gt_bboxes)), matches[:, 1])
        
        if len(unmatched_gt_indxs) != 0:
            all_ious.append(np.zeros(len(unmatched_gt_indxs)))
            all_coverage.append(np.zeros(len(unmatched_gt_indxs)))
            all_sharpness.append(np.zeros(len(unmatched_gt_indxs)))
        
        unmatched_pred_indxs = np.setdiff1d(np.arange(len(probs)), matches[:, 0])
        
        if len(unmatched_pred_indxs) != 0:
            all_ious.append(np.zeros(len(unmatched_pred_indxs)))
            all_coverage.append(np.zeros(len(unmatched_pred_indxs)))
            all_sharpness.append(np.zeros(len(unmatched_pred_indxs)))
        continue
    


    iou_array, coverage_array, sharpness_array = get_iou_coverage_sharpness(old_bboxes=gt_bboxes[matches[:, 1]], new_bboxes=bbox_preds[matches[:, 0]])
    all_ious.append(iou_array)
    all_coverage.append(coverage_array)
    all_sharpness.append(sharpness_array)
    matched_ious.append(iou_array)
    matched_coverage.append(coverage_array)
    matched_sharpness.append(sharpness_array)

    if len(bbox_preds) - len(matches) > 0: 
        all_ious.append(np.zeros(len(bbox_preds) - len(matches)))
        all_coverage.append(np.zeros(len(bbox_preds) - len(matches)))
        all_sharpness.append(np.zeros(len(bbox_preds) - len(matches)))


all_ious = np.concatenate(all_ious)
all_coverage = np.concatenate(all_coverage)
all_sharpness = np.concatenate(all_sharpness)

matched_ious = np.concatenate(matched_ious)
matched_coverage = np.concatenate(matched_coverage)
matched_sharpness = np.concatenate(matched_sharpness)

results_row = [args.score_threshold, args.iou_threshold, all_ious.mean(), all_coverage.mean(), all_sharpness.mean(), matched_ious.mean(), matched_coverage.mean(), matched_sharpness.mean()]

print("Results:", results_row)

if args.result_output_filename is not None:
    with open(args.result_output_filename, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results_row)