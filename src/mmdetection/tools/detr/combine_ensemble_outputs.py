import numpy as np
import torch
import mmdet
from mmdet.apis import init_detector
from mmdet.datasets import build_dataset, build_dataloader
from tqdm import tqdm #progress bar
import pickle
import sys
import os


base_file_name = ""
epochs = [i for i in range(1, 30)]

final_output = None
for epoch in epochs:
    with open(os.path.join(base_file_name, 'output_epoch_{}.pkl'.format(epoch)), 'rb') as f:
        results = pickle.load(f)
    if final_output is None:
        final_output = results
        bbox_preds = [[result['bbox_preds'] for result in results]]
        cls_probs = [[result['cls_probs'] for result in results]]
    else:
        bbox_preds.append([result['bbox_preds'] for result in results])
        cls_probs.append([result['cls_probs'] for result in results])

mean_bbox_preds = np.array(bbox_preds).squeeze().mean(axis=0)
mean_cls_probs = np.array(cls_probs).squeeze().mean(axis=0)

std_bbox_preds = np.std(np.array(bbox_preds).squeeze(), axis=0)


for i, result in enumerate(final_output):
    result['bbox_preds'] = mean_bbox_preds[i]
    result['cls_probs'] = mean_cls_probs[i]
    result['std_bbox_preds'] = std_bbox_preds[i]

with open(os.path.join(base_file_name, 'combined_output.pkl'), 'wb') as f:
    pickle.dump(final_output, f, protocol=pickle.HIGHEST_PROTOCOL)
