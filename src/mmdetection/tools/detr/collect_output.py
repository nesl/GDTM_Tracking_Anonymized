import numpy as np
import torch
import mmdet
from mmdet.apis import init_detector
from mmdet.datasets import build_dataset, build_dataloader
from tqdm import tqdm, trange#progress bar
import pickle
import sys

#init model and dataloader
config = sys.argv[1]
checkpoint = sys.argv[2]
pkl_fname = sys.argv[3]
model = init_detector(config, checkpoint).cuda().eval()
dataset = build_dataset(model.cfg.data.val)
# dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

output = []

for idx in trange(len(dataset)):
    sample = dataset[idx]
    gt = dataset.get_ann_info(idx)
    img = sample['img'][0].cuda().unsqueeze(0)
    img_metas = [sample['img_metas'][0].data]
    img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
    result = {k: v for k, v in img_metas[0].items()}

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

        if model.bbox_head.count_dist == 'normal':
            count_preds = model.bbox_head.activate(model.bbox_head.count_ffn(query_embeds))
            count_preds = model.bbox_head.fc_count(count_preds)
            import ipdb; ipdb.set_trace() # noqa
            count_preds = count_preds.mean(dim=-2) #b x Ndec x 80*2
            mean, var = count_preds.split(80, dim=-1)
            var = model.bbox_head.softplus(var)
            result['mean'] = mean.cpu().numpy()
            result['var'] = var.cpu().numpy()
            # model.bbox_head.count_dists = Normal(mean, var)

        elif model.bbox_head.count_dist == 'poisson':
            count_preds = model.bbox_head.activate(model.bbox_head.count_ffn(query_embeds))
            count_preds = model.bbox_head.fc_count(count_preds)
            count_preds = count_preds.mean(dim=2) #b x Ndec x 80*2
            count_rates = model.bbox_head.softplus(count_preds) #stash here for now
            result['count_rates'] = count_rates.cpu().numpy()
            # model.bbox_head.count_dists = Poisson(count_rates)


        # if model.bbox_head.loss_count is not None:
            # count_preds = model.bbox_head.count_ffn(query_embeds) #linear
            # count_preds = model.bbox_head.activate(count_preds) #relu
            # count_preds = model.bbox_head.fc_count(count_preds) #linear
            # count_preds = count_preds.mean(dim=1)
            # count_rates = model.bbox_head.softplus(count_preds)

    
    #collect output
    result['query_embeds'] = query_embeds.cpu().numpy()
    result['bbox_preds'] = bbox_preds.cpu().numpy()
    result['cls_probs'] = cls_probs.cpu().numpy()

    result['gt_bboxes'] = gt['bboxes']
    result['gt_labels'] = gt['labels']
    output.append(result)
    
#save to pickle file
with open(pkl_fname, 'wb') as f:
    pickle.dump(output, f)
