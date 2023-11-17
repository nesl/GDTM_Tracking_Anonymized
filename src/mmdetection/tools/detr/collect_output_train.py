import numpy as np
import torch
import mmdet
from mmdet.apis import init_detector
from mmdet.datasets import build_dataset, build_dataloader
from tqdm import tqdm, trange#progress bar
import pickle
import sys
import argparse

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# parser = argparse.ArgumentParser(description='collect_output.py')
# parser.add_argument('--mode', default='val')
# args = vars(parser.parse_args())

#init model and dataloader
config = sys.argv[1]
checkpoint = sys.argv[2]
pkl_fname = sys.argv[3]
model = init_detector(config, checkpoint).cuda().eval()
# dataset_cfg = model.cfg.data.val

# collect_cfg = model.cfg.data.train.pipeline[-1]
# model.cfg.data.train['pipeline'] = model.cfg.data.val['pipeline']
# model.cfg.data.train.pipeline[-1]['transforms'][-1] = collect_cfg
dataset_cfg = model.cfg.data.train
dataset = build_dataset(dataset_cfg)
dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False, seed=1)

output = []

#for idx in trange(len(dataset)):
for idx, sample in enumerate(tqdm(dataloader)):
    # print(idx)
    if idx % 30000 == 0 and idx != 0:
        with open(pkl_fname, 'wb') as f:
            pickle.dump(output, f)
    img = sample['img'].data[0].cuda()
    img_metas = sample['img_metas'].data[0]
    # sample = dataset[idx]
    # gt = dataset.get_ann_info(idx)
    # img = sample['img'][0].cuda().unsqueeze(0)
    # img_metas = [sample['img_metas'][0].data]
    img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
    result = {k: v for k, v in img_metas[0].items()}
    
    result['gt_bboxes'] = sample['gt_bboxes'].data[0][0].numpy()
    result['gt_labels'] = sample['gt_labels'].data[0][0].numpy()
    # if len(result['gt_labels']) == 0:
        # import pdb
        # pdb.set_trace()

    with torch.no_grad():
        #run backbone and transformer
        feats = model.extract_feats([img])[0] #cnn
        query_embeds = model.bbox_head.forward_single(feats[0], img_metas)
        query_embeds = query_embeds.squeeze() #6 x 100 x 256
        
        #predict softmax dists
        cls_scores = model.bbox_head.fc_cls(query_embeds) #linear
        cls_probs = torch.softmax(cls_scores, dim=-1) #6 x 100 x 81
        
        #predict bboxes
        bbox_preds = model.bbox_head.reg_ffn(query_embeds) #linear
        bbox_preds = model.bbox_head.activate(bbox_preds) #relu
        bbox_preds = model.bbox_head.fc_reg(bbox_preds) #linear
        bbox_preds = torch.sigmoid(bbox_preds) #6 x 100 x 4
        
        result['bbox_preds'] = bbox_preds.cpu()[-1].unsqueeze(0).numpy()
        result['cls_probs'] = cls_probs.cpu()[-1].unsqueeze(0).numpy()

        # result['query_embeds'] = query_embeds.cpu().numpy()
        # result['bbox_preds'] = bbox_preds.cpu().numpy()
        # result['cls_probs'] = cls_probs.cpu().numpy()

            
    
    #collect output
    output.append(result)
    
#save to pickle file
with open(pkl_fname, 'wb') as f:
    pickle.dump(output, f)
