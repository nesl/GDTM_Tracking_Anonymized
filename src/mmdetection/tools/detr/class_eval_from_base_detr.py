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

parser = argparse.ArgumentParser()

parser.add_argument("--gt_pkl_filename", help="Pickle file containing ground truth.",
                    type=str)
parser.add_argument('--result_output_filename', help="Output CSV filename", 
                    type=str)
parser.add_argument('--score_threshold', help='Minimum score threshold for predictions', 
                    type=float)
parser.add_argument('--iou_threshold', help='Minimum IoU threshold for matching', 
                    type=float, default=0.5)
parser.add_argument('--result_identifiers', type=str, help='Additional identifiers for CSV', default=None)


args = parser.parse_args()


def _get_ece(preds, targets, n_bins=15):
    """
    ECE ported from Asukha et al., 2020.
    :param preds: Prediction probabilities in a Numpy array
    :param targets: Targets in a numpy array
    :param n_bins: Total number of bins to use.
    :return: Expected calibration error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = (predictions == targets)

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece

def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    matches = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(matches[:, 1])
    matches = matches[sort_idx]
    matches = torch.from_numpy(matches).long()
    return matches



with open(args.gt_pkl_filename, 'rb') as f:
    data = pickle.load(f)
    
all_probs, all_labels = [], []
matched_probs, matched_labels = [], []
count_unmatched = 0
count_no_preds = 0
total_gt_bboxes = 0

IOU_THRESHOLD = args.iou_threshold

for sample in data:
    gt_bboxes = torch.from_numpy(sample['gt_bboxes'])
    gt_labels = torch.from_numpy(sample['gt_labels']).long()    
    probs = torch.from_numpy(sample['cls_probs'])[-1]
    is_bg = (torch.argmax(probs, dim=-1) == (probs.shape[1] - 1)) 
    max_probs, _ = probs.max(-1)
    is_conf = max_probs >= args.score_threshold

    mask = ~is_bg & is_conf
    probs = probs[mask]
    
    #scale and convert bbox preds to match gt format
    H, W, _ = sample['ori_shape']
    factor = torch.tensor([W, H, W, H]).unsqueeze(0)
    bbox_preds = torch.from_numpy(sample['bbox_preds'][-1]) 
    bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds) * factor
    
    bbox_preds = bbox_preds[mask]
    
    if len(bbox_preds) == 0:
        count_no_preds += 1
        count_unmatched += len(gt_labels)
        all_labels.append(gt_labels)
        all_probs.append(torch.ones(len(gt_labels), 81)/81.)
        continue
    
    #compute pairwise iou between all predictions and gt
    #matrix has shape N_preds x N_gt (for detr, N_preds == 100)
    ious = BboxOverlaps2D()(bbox_preds, gt_bboxes)
    
    #set iou to 0 for all boxes predicted as background (optional?)
     
    #ious[bg_mask] = 0
    
    #run min cost assignment algo using negative iou 
    #matrix has shape N_gt x 2
    original_matches = matches = linear_assignment(1-ious.numpy())
    sel_indices = ious.numpy()[matches.numpy()[:, 0], matches.numpy()[:, 1]] >= IOU_THRESHOLD
    matches = matches[sel_indices]
    
    #all gt boxes should be matched to something (could be background)
    #assert len(matches) == len(gt_bboxes)
    if len(matches) != len(gt_bboxes):
        if len(gt_bboxes) > len(matches): 
            count_unmatched += len(gt_bboxes) - len(matches)
        matched_label_targets = torch.ones(len(matches)).long() * 80
        matched_label_targets = gt_labels[matches[:, 1]]
        all_probs.append(probs[matches[:, 0]])
        all_labels.append(matched_label_targets)
        matched_probs.append(probs[matches[:, 0]])
        matched_labels.append(matched_label_targets)
        unmatched_gt_indxs = np.setdiff1d(np.arange(len(gt_bboxes)), matches[:, 1])
        
        if len(unmatched_gt_indxs) != 0:
            all_labels.append(gt_labels[unmatched_gt_indxs])
            all_probs.append(torch.ones(len(unmatched_gt_indxs), 81)/81.)
        
        unmatched_pred_indxs = np.setdiff1d(np.arange(len(probs)), matches[:, 0])
        if len(unmatched_pred_indxs) != 0:
            all_labels.append(torch.ones(len(unmatched_pred_indxs)).long() * 80)
            all_probs.append(probs[torch.from_numpy(unmatched_pred_indxs)])
        continue
    
    #convert full label targets
    #most predictions are unmatched and are therefore background
    label_targets = torch.ones(len(bbox_preds)).long() * 80 #background
    label_targets[matches[:, 0]] = gt_labels #gt_labels for matched predictions
    
    #save all probs and assigned labels
    all_probs.append(probs)
    all_labels.append(label_targets)

    
all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)
matched_probs = torch.cat(matched_probs)
matched_labels = torch.cat(matched_labels)


def get_nll(probs, labels):
    nll_vals = []
    for i in range(len(probs)):
        p = probs[i]
        l = labels[i]
        nll_vals.append(-torch.log(p[l]))
    nll_vals = torch.tensor(nll_vals)
    return nll_vals.mean().item()

def get_acc(probs, labels):
    _, max_idx = probs.max(-1)
    acc = (max_idx == labels).float().mean()
    return acc.item()

ece = _get_ece(all_probs.numpy(), all_labels.numpy())
nll = get_nll(all_probs, all_labels)
acc = get_acc(all_probs, all_labels)

ece_matched = _get_ece(matched_probs.numpy(), matched_labels.numpy())
nll_matched = get_nll(matched_probs, matched_labels)
acc_matched = get_acc(matched_probs, matched_labels)

results_row = [args.score_threshold, args.iou_threshold, ece, nll, acc, ece_matched, nll_matched, acc_matched, len(matched_probs), count_unmatched, len(all_labels), len(all_probs)]

if args.result_identifiers is not None:
    result_identifiers = args.result_identifiers.split(",")
    results_row = results_row + result_identifiers

print(results_row)

with open(args.result_output_filename, 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(results_row)