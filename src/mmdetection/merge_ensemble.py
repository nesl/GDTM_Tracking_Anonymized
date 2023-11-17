import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from tqdm import tqdm

import numpy as np
# import tqdm
import json
import sys
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sampling_detector import SamplingDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Test the data and save the raw detections')
    # parser.add_argument('--dataset', default = 'voc', help='voc or coco')
    # parser.add_argument('--subset', default = None, help='train or val or test')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    parser.add_argument('--iou', default = 0.8, type = float, help='bbox iou to merge ensemble detections')
    parser.add_argument('--num_models', default=None, type=int, help='num of models in ensemble')
    parser.add_argument('--maximal', action='store_true', help='generate a maximal bounding box')
    parser.add_argument('--minimal', action='store_true', help='generate a minimal bounding box')
    parser.add_argument('--save_dir', default=None, help='Path to load models from')
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

num_models = args.num_models

#used to take in collection of individual ensemble results and convert into merged ensemble results
merger = SamplingDetector(iou = args.iou, min_dets=6)

allOutputs = [None for i in range(num_models)]
#load results from each individual model
# save_dir = 'logs/detr_r50_4x16_decoder_and_output'
# save_dir = 'logs/detr_r50_8x2_150e_coco'
save_dir = args.save_dir
for i in range(1, num_models+1):
    try:
        print("Trying to load from path:" + f'{save_dir}/{args.saveNm}_{i}.json')
        with open(f'{save_dir}/{args.saveNm}_{i}.json', 'r') as f:
            allOutputs[i-1] = json.load(f)
    except:
        print(f'Missing results file for {save_dir}/{args.saveNm}_{i}.json')
        exit()

ensembleResults = {}
time_array = []
for imIdx, imKey in enumerate(tqdm(allOutputs[0].keys())):
    
    ensembleResults[imKey] = []
    #collect all detections for this image
    ensemble_detections = []
    for ensIdx in range(num_models):
        detections = np.array(allOutputs[ensIdx][imKey])

        if len(detections) == 0: #no detections
            continue

        if len(ensemble_detections) == 0:
            ensemble_detections = detections
        else:
            ensemble_detections = np.concatenate((ensemble_detections, detections), axis = 0)

    if len(ensemble_detections) == 0:
        continue
    
    t0 = time.perf_counter()
    #cluster and merge ensemble detections into final detections (don't pass in final column with softmax score)
    clustering_output = merger.form_final(ensemble_detections[:, :-1], maximal=args.maximal, minimal=args.minimal, return_clusters=True)
    t1 = time.perf_counter()
    time_array.append(t1-t0)

    if len(clustering_output) == 0: #no valid detections were clustered
        continue
    
    final_detections, clusters = clustering_output[0], clustering_output[1]

    #calculate new max softmax score and concatenate to detections
    distsT = torch.Tensor(final_detections[:, :-4])
    # softmaxScores = softmaxLayer(distsT).numpy()
    scores = np.max(distsT.numpy(), axis = 1)
    scoresT = np.expand_dims(scores, axis=1)


    imDets = np.concatenate((final_detections, scoresT), 1)
    ensembleResults[imKey] = [imDets.tolist(), clusters]


#save results
jsonRes = json.dumps(ensembleResults, indent=4)
time_array = np.array(time_array)
print("Median clustering time: ", np.median(time_array))
print("Mean clustering time: ", np.mean(time_array))

# save_dir = f'{save_dir}/FRCNN/raw/{args.dataset}/{args.subset}'
iouEns = str(args.iou).replace('.', '')
f = open(f'{save_dir}/merged_outputs_{iouEns}_{args.num_models}_ensemble_maximal.json', 'w')
f.write(jsonRes)
f.close()
