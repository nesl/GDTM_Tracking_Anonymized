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
parser.add_argument('--merged_outputs_filename', help="JSON file for the merged outputs.",
                    type=str)
parser.add_argument('--result_output_filename', help="Output CSV filename", 
                    type=str)
parser.add_argument('--score_threshold', help="Score threshold used in filtering.", 
                    type=float)

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

with open(args.merged_outputs_filename) as f:
    merged_outputs = json.load(f)
    
with open(args.gt_pkl_filename, 'rb') as f:
    model_output = pickle.load(f)
    

fnames = [o['ori_filename'] for o in model_output]


all_probs, all_labels = [], []
all_bbox_preds, all_bbox_targets = [], []
all_bbox_clusters = []
all_bbox_h_w = []
count_unmatched = 0
count_no_preds = 0
total_gt_bboxes = 0
unmatched_gt_boxes = []

IOU_THRESHOLD = 0.5

EMPIRICAL_VAR_X = 0.0001555555556 #this is for full ensemble
# EMPIRICAL_VAR_X = 0.0002 #this is for ensemble
EMPIRICAL_VAR_Y = 0.0002 #this is for ensemble
counter = 0.

for sample in model_output:
    fname = sample['ori_filename']
    H, W, _ = sample['ori_shape']
    gt_bboxes = torch.from_numpy(sample['gt_bboxes'])
    gt_labels = torch.from_numpy(sample['gt_labels']).long()    
    sample_bbox_clusters = []
    
    
    total_gt_bboxes += len(gt_labels)
    if merged_outputs[fname] == []:
        continue

    preds = merged_outputs[fname][0]
    clusters = merged_outputs[fname][1]
    preds = torch.tensor(preds)
    if len(preds) == 0:
        count_no_preds += 1
        #print(preds, fname, len(gt_labels))
        all_labels.append(gt_labels)
        all_probs.append(torch.ones(len(gt_labels), 81)/81.)
        continue
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
    # clusters = np.array(clusters)[mask.numpy()].tolist()
    clusters = np.array(clusters)[mask.numpy()]

    if len(bbox_preds) == 0:
        count_no_preds += 1
        all_labels.append(gt_labels)
        all_probs.append(torch.ones(len(gt_labels), 81)/81.)
        continue
        
    #scale and convert bbox preds to match gt format
    #H, W, _ = sample['ori_shape']
    #factor = torch.tensor([W, H, W, H]).unsqueeze(0)
    #bbox_preds = torch.from_numpy(sample['bbox_preds'][-1]) 
    #bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds) * factor
    
    #compute pairwise iou between all predictions and gt
    #matrix has shape N_preds x N_gt (for detr, N_preds == 100)
    ious = BboxOverlaps2D()(bbox_preds, gt_bboxes)
    gt_bboxes_centres = bbox_xyxy_to_cxcywh(gt_bboxes)[:, :2].div(torch.tensor([W, H]))
    pred_bboxes_centres_mean = bbox_xyxy_to_cxcywh(bbox_preds)[:, :2]
    # pred_bboxes_centres_var = [bbox_xyxy_to_cxcywh(torch.tensor(cluster)[:, -4:]).div(torch.tensor([W, H, W, H])).var(dim=0) for cluster in clusters]
    # pred_bboxes_centres_var = torch.stack(pred_bboxes_centres_var)[:, :2]
    pred_bboxes_centres_var = torch.ones(len(pred_bboxes_centres_mean), 2)
    pred_bboxes_centres_var[:, 0], pred_bboxes_centres_var[:, 1] = pred_bboxes_centres_var[:, 0] * EMPIRICAL_VAR_X, pred_bboxes_centres_var[:, 1] * EMPIRICAL_VAR_Y 
    pred_bboxes_centres_mean[:, 0], pred_bboxes_centres_mean[:, 1] = pred_bboxes_centres_mean[:, 0]/W, pred_bboxes_centres_mean[:, 1]/H
    pred_bboxes_centres_std = torch.sqrt(pred_bboxes_centres_var)


    #set iou to 0 for all boxes predicted as background (optional?)
    #bg_mask = probs[:, -1] >= 0.5 
    #ious[bg_mask] = 0
    
    #run min cost assignment algo using negative iou 
    #matrix has shape N_gt x 2
    matches = linear_assignment(1.-ious.numpy())
    sel_indices = ious.numpy()[matches.numpy()[:, 0], matches.numpy()[:, 1]] >= IOU_THRESHOLD
    matches = matches[sel_indices]
    
    #all gt boxes should be matched to something (could be background)
    if len(matches) != len(gt_bboxes):
        # count_unmatched += len(gt_bboxes) - len(bbox_preds)
        # matched_label_targets = torch.ones(len(bbox_preds)).long() * 80
        # matched_bbox_centre_targets = torch.zeros(len(bbox_preds), 2)   #torch.ones(len(bbox_preds)).long() * 80
        # matched_label_targets[matches[:, 0]] = gt_labels[matches[:, 1]]
        # matched_bbox_centre_targets[matches[:, 0]] = gt_bboxes_centres[matches[:, 1]]
        # all_probs.append(probs)
        # all_labels.append(matched_label_targets)
        # all_bbox_preds.append(torch.cat([pred_bboxes_centres_mean, pred_bboxes_centres_std], dim=1))
        # all_bbox_targets.append(matched_bbox_centre_targets)
        # unmatched_gt_indxs = np.setdiff1d(np.arange(len(gt_bboxes)), matches[:, 1])
        # all_labels.append(gt_labels[unmatched_gt_indxs])
        # all_probs.append(torch.ones(len(unmatched_gt_indxs), 81)/81.)
        # continue

        count_unmatched += len(gt_bboxes) - len(matches)
        matched_label_targets = torch.ones(len(matches)).long() * 80
        matched_label_targets = gt_labels[matches[:, 1]]
        all_probs.append(probs[matches[:, 0]])
        all_labels.append(matched_label_targets)
        matched_bbox_centre_targets = torch.zeros(len(matches), 2)
        matched_bbox_centre_targets = gt_bboxes_centres[matches[:, 1]]
        all_bbox_h_w.append(torch.ones(len(matched_bbox_centre_targets), 4) * torch.tensor([W, H, W, H]))
        counter += len(matches)

        # if counter >= 49:
        #     import pdb
        #     pdb.set_trace()

        if type(clusters[matches[:, 0]]) !=list and (len(clusters[matches[:, 0]].shape) == 1 or len(clusters[matches[:, 0]].shape) == 3):
            all_bbox_clusters += clusters[matches[:, 0]].tolist()
        else:
            all_bbox_clusters += np.array([clusters[matches[:, 0]]], dtype='object').tolist()

        all_bbox_preds.append(torch.cat([pred_bboxes_centres_mean, pred_bboxes_centres_std], dim=1)[matches[:, 0]])
        all_bbox_targets.append(matched_bbox_centre_targets)
        unmatched_gt_indxs = np.setdiff1d(np.arange(len(gt_bboxes)), matches[:, 1])
        if len(unmatched_gt_indxs) != 0:
            all_labels.append(gt_labels[unmatched_gt_indxs])
            all_probs.append(torch.ones(len(unmatched_gt_indxs), 81)/81.)
        
        unmatched_pred_indxs = np.setdiff1d(np.arange(len(probs)), matches[:, 0])
        if len(unmatched_pred_indxs) != 0:
            all_labels.append(torch.ones(len(unmatched_pred_indxs)).long() * 80)
            all_probs.append(probs[torch.from_numpy(unmatched_pred_indxs)])
        continue
        #print(len(matches), len(gt_bboxes), len(bbox_preds), len(gt_bboxes))
    
    #convert full label targets
    #most predictions are unmatched and are therefore background
    label_targets = torch.ones(len(bbox_preds)).long() * 80 #background
    matched_bbox_centre_targets = torch.zeros(len(bbox_preds), 2)   #torch.ones(len(bbox_preds)).long() * 80
    label_targets[matches[:, 0]] = gt_labels #gt_labels for matched predictions
    matched_bbox_centre_targets[matches[:, 0]] = gt_bboxes_centres
    
    #save all probs and assigned labels
    all_probs.append(probs)
    all_labels.append(label_targets)

    
    pred_order, _ = torch.sort(matches[:, 0])
    all_bbox_targets.append(matched_bbox_centre_targets[pred_order])
    all_bbox_h_w.append(torch.ones(len(matched_bbox_centre_targets[pred_order]), 4) * torch.tensor([W, H, W, H]))
    counter += len(gt_bboxes)
    # if counter >= 49:
    #     import pdb
    #     pdb.set_trace()
    if type(clusters[pred_order]) != list and (len(clusters[pred_order].shape) == 1 or len(clusters[pred_order].shape) == 3):
        all_bbox_clusters += clusters[pred_order].tolist()
    else:
        # import pdb
        # pdb.set_trace()
        all_bbox_clusters += np.array([clusters[pred_order]], dtype='object').tolist()
    # all_bbox_clusters += clusters[pred_order]
    # all_bbox_clusters.append(sample_bbox_clusters)
    all_bbox_preds.append(torch.cat([pred_bboxes_centres_mean, pred_bboxes_centres_std], dim=1)[pred_order])

    
all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)
all_bbox_targets = torch.cat(all_bbox_targets)
all_bbox_preds = torch.cat(all_bbox_preds)
all_bbox_h_w = torch.cat(all_bbox_h_w)

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

# def get_centre_log_loss(bbox_preds, bbox_targets):
#     nll_vals = []
#     for i in range(len(bbox_targets)):
#         bbox_target = bbox_targets[i]
#         bbox_means = bbox_preds[i, :2]
#         bbox_std = bbox_preds[i, 2:]
#         d = Normal(loc=bbox_means, scale=bbox_std)
#         nll_vals.append(-d.log_prob(bbox_target).sum())
#     nll_vals = torch.tensor(nll_vals)
#     return nll_vals.mean().item()

def get_centre_log_loss(clusters, bbox_targets, bbox_h_w):
    nll_vals = []
    for i in range(len(bbox_targets)):
        cluster = clusters[i]
        pred_bboxes_centres = bbox_xyxy_to_cxcywh(torch.tensor(cluster)[:, -4:]).div(bbox_h_w[i])[:, :2]
        pred_bboxes_centres_var = torch.ones(len(pred_bboxes_centres), 2)
        pred_bboxes_centres_var[:, 0], pred_bboxes_centres_var[:, 1] = pred_bboxes_centres_var[:, 0] * EMPIRICAL_VAR_X, pred_bboxes_centres_var[:, 1] * EMPIRICAL_VAR_Y 
        pred_bboxes_centres_std = torch.sqrt(pred_bboxes_centres_var)
        bbox_target = bbox_targets[i]
        per_member_proba = []
        for j in range(len(pred_bboxes_centres)):
            d = Normal(loc=pred_bboxes_centres[j], scale=pred_bboxes_centres_std[j])
            # per_member_proba.append(d.log_prob(bbox_target).sum().exp())
            per_member_proba.append(d.log_prob(bbox_target).sum() + torch.log(torch.tensor(1/len(pred_bboxes_centres)))) #Log probs with equal weight to all members of the cluster
        # import pdb
        # pdb.set_trace()
        # nll_vals.append(-torch.log(torch.tensor(per_member_proba).mean()))
        nll_vals.append(-torch.logsumexp(torch.tensor(per_member_proba), 0)) #Logsumexp is more stable for computing NLL
    nll_vals = torch.tensor(nll_vals)
    inf_idxs = torch.isinf(nll_vals)
    # import pdb
    # pdb.set_trace()
    return nll_vals.mean().item()

bbox_centre_nll = get_centre_log_loss(all_bbox_clusters, all_bbox_targets, all_bbox_h_w)

results_row = [args.score_threshold, bbox_centre_nll]

print("Results:", results_row)

if args.result_output_filename is not None:
    with open(args.result_output_filename, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results_row)