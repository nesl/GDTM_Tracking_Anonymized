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
import math
import skimage.measure
from torch.distributions.normal import Normal
from sklearn import metrics

parser = argparse.ArgumentParser()

parser.add_argument("--gt_pkl_filename", help="Pickle file containing ground truth.",
                    type=str)
parser.add_argument('--merged_outputs_filename', help="JSON file for the merged outputs.",
                    type=str)
parser.add_argument('--score_threshold', help="Score threshold used in merging.", 
                    type=float, default=None)


args = parser.parse_args()


with open(args.merged_outputs_filename) as f:
    merged_outputs = json.load(f)
    
with open(args.gt_pkl_filename, 'rb') as f:
    model_output = pickle.load(f)
    

fnames = [o['ori_filename'] for o in model_output]    

all_ground_truth_maps = []
all_prediction_maps = []

max_pool_kernel_shape = (4, 4)

for sample in model_output:
    fname = sample['ori_filename']
    H, W, _ = sample['ori_shape']
    gt_bboxes = torch.from_numpy(sample['gt_bboxes'])
    gt_labels = torch.from_numpy(sample['gt_labels']).long()
    gt_bboxes = gt_bboxes.numpy()
    
    ground_truth_map = []
    for i, gt_bbox in enumerate(gt_bboxes):
        gt_map = np.zeros((W, H))
        gt_map[math.floor(gt_bbox[0]): math.ceil(gt_bbox[2]), math.floor(gt_bbox[1]):math.ceil(gt_bbox[3])] = 1
        gt_map = skimage.measure.block_reduce(gt_map, max_pool_kernel_shape, np.max)
        ground_truth_map.append(gt_map)
    ground_truth_map = np.max(np.stack(ground_truth_map), axis=0)
    all_ground_truth_maps.append(ground_truth_map.reshape(-1, ))

    if merged_outputs[fname] == []:
        final_pred_map = np.zeros((W, H))
        final_pred_map = skimage.measure.block_reduce(final_pred_map, max_pool_kernel_shape, np.max)
        all_prediction_maps.append(final_pred_map.reshape(-1, ))
        continue
    
    preds = merged_outputs[fname][0]
    preds = torch.tensor(preds)

    probs = preds[:, 0:80]
    sums = probs.sum(dim=-1).unsqueeze(-1)
    bg_probs = torch.ones_like(sums) - sums
    probs = torch.cat([probs, bg_probs], dim=-1)
    is_bg = (torch.argmax(probs, dim=-1) == (probs.shape[1] - 1))
    max_probs, _ = probs.max(-1)
    is_conf = max_probs >= args.score_threshold
    mask = ~is_bg & is_conf
    probs = probs[mask]
    
    bbox_preds = preds[:, -5:-1]
    bbox_preds = bbox_preds[mask]
    bbox_preds = bbox_preds.numpy()
    bbox_preds = bbox_preds.clip(min=0)
    probs = probs.numpy()
    
    final_pred_map = []
    
    if len(bbox_preds) == 0:
        final_pred_map = np.zeros((W, H))
        final_pred_map = skimage.measure.block_reduce(final_pred_map, max_pool_kernel_shape, np.max)
        all_prediction_maps.append(final_pred_map.reshape(-1, ))
        continue
    
    for i, pred_bbox in enumerate(bbox_preds):
        pred_map = np.zeros((W, H)) 
        pred_map[math.floor(pred_bbox[0]): math.ceil(pred_bbox[2]), math.floor(pred_bbox[1]): math.ceil(pred_bbox[3])] = 1 - probs[i, -1]
        # pred_map = skimage.measure.block_reduce(pred_map, max_pool_kernel_shape, np.max)
        final_pred_map.append(pred_map)
    final_pred_map = np.max(np.stack(final_pred_map), axis=0)
    final_pred_map = skimage.measure.block_reduce(final_pred_map, max_pool_kernel_shape, np.max)
    # import pdb
    # pdb.set_trace()
    all_prediction_maps.append(final_pred_map.reshape(-1, ))

pixel_pred_array = np.hstack(all_prediction_maps)
pixel_gt_array = np.hstack(all_ground_truth_maps)

nll = metrics.log_loss(pixel_gt_array, pixel_pred_array, eps=1e-3)
accuracy = metrics.accuracy_score(pixel_gt_array, (pixel_pred_array>=0.5).astype(int))
auroc = metrics.roc_auc_score(pixel_gt_array, pixel_pred_array)
precision = metrics.precision_score(pixel_gt_array, (pixel_pred_array>=0.5).astype(int))
recall = metrics.recall_score(pixel_gt_array, (pixel_pred_array>=0.5).astype(int))
f1 = metrics.f1_score(pixel_gt_array, (pixel_pred_array>=0.5).astype(int))

print(nll, accuracy, auroc, precision, recall, f1)