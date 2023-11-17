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

parser.add_argument('--gt_pkl_filename', help="Pickle file containing ground truth.",
                    type=str)
parser.add_argument('--ensemble_output_dir', help="Directory containing bbox outputs for each member of the ensemble.",
                    type=str)
parser.add_argument('--bbox_filename', help='Filename template for bounding box outputs', type=str)
parser.add_argument('--ensemble_size', help='Num of members in the ensemble', type=int)
parser.add_argument('--score_threshold', help="Score threshold used in merging.", 
                    type=float, default=None)


args = parser.parse_args()
    
with open(args.gt_pkl_filename, 'rb') as f:
    model_output = pickle.load(f)

ensemble_output_list = []
for i in range(args.ensemble_size):
    with open(os.path.join(args.ensemble_output_dir, "%s_%d.json" % (args.bbox_filename, i+1)), 'r') as f:
        ensemble_output_list.append(json.load(f))

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

    image_output_list = []
    image_ensemble_index_list = []
    for j in range(args.ensemble_size):
        if ensemble_output_list[j][fname] != []:
            image_output_list.append(np.array(ensemble_output_list[j][fname]))
            image_ensemble_index_list.append(np.ones(len(ensemble_output_list[j][fname])) * j)
    
    if image_output_list == []:
        final_pred_map = np.zeros((W, H))
        final_pred_map = skimage.measure.block_reduce(final_pred_map, max_pool_kernel_shape, np.max)
        all_prediction_maps.append(final_pred_map.reshape(-1, ))
        continue
    else:
        preds = np.vstack(image_output_list)
        image_ensemble_index_list = np.hstack(image_ensemble_index_list)
    
    # preds = merged_outputs[fname][0]
    preds = torch.from_numpy(preds)

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
    try:
        image_ensemble_index_list = image_ensemble_index_list[mask]
    except:
        image_ensemble_index_list = image_ensemble_index_list[mask.numpy()]
    probs = probs.numpy()
    
    final_pred_map = []
    
    if len(bbox_preds) == 0:
        final_pred_map = np.zeros((W, H))
        final_pred_map = skimage.measure.block_reduce(final_pred_map, max_pool_kernel_shape, np.max)
        all_prediction_maps.append(final_pred_map.reshape(-1, ))
        continue
    
    for j in range(args.ensemble_size):
        model_pred_map = []
        if len(bbox_preds[image_ensemble_index_list==j]) > 0:
            model_bbox_preds = bbox_preds[image_ensemble_index_list==j].reshape(-1, 4)
            model_probs = probs[image_ensemble_index_list==j].reshape(-1, 81)
            for i, pred_bbox in enumerate(model_bbox_preds):
                pred_map = np.zeros((W, H)) 
                pred_map[math.floor(pred_bbox[0]): math.ceil(pred_bbox[2]), math.floor(pred_bbox[1]): math.ceil(pred_bbox[3])] = 1 - model_probs[i, -1]
                # pred_map = skimage.measure.block_reduce(pred_map, max_pool_kernel_shape, np.max)
                model_pred_map.append(pred_map)
            model_pred_map = np.max(np.stack(model_pred_map), axis=0)
            final_pred_map.append(model_pred_map)
    final_pred_map = np.mean(np.stack(final_pred_map), axis=0)
    final_pred_map = skimage.measure.block_reduce(final_pred_map, max_pool_kernel_shape, np.max)
    # if fname=='000000504711.jpg':
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
