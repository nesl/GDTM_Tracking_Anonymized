import numpy as np
import torch
import torch.nn.functional as F
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

times = {'backbone': [], 'encoder': [], 'decoder': [], 'output_head': []}
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for idx in trange(len(dataset)):
    sample = dataset[idx]
    gt = dataset.get_ann_info(idx)
    img = sample['img'][0].cuda().unsqueeze(0)
    img_metas = [sample['img_metas'][0].data]
    img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
    result = {k: v for k, v in img_metas[0].items()}
    
    with torch.no_grad():
        start.record()

        x = model.extract_feats([img])[0][0] #cnn
        x = model.bbox_head.input_proj(x)

        end.record()
        torch.cuda.synchronize()
        result['backbone_time'] = start.elapsed_time(end) ##########
        
        start.record() ###############################################
        
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = model.bbox_head.positional_encoding(masks)  # [bs, embed_dim, h, w]
        query_embed = model.bbox_head.query_embedding.weight
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        masks = masks.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = model.bbox_head.transformer.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=masks)
        
        end.record()
        torch.cuda.synchronize()
        result['encoder_time'] = start.elapsed_time(end) ##########
        
        start.record() ###############################################

        target = torch.zeros_like(query_embed)
        out_dec = model.bbox_head.transformer.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=masks)
        query_embeds = out_dec.transpose(1, 2)

        end.record()
        torch.cuda.synchronize()
        # times['decoder'].append(start.elapsed_time(end)) ##########
        result['decoder_time'] = start.elapsed_time(end) ##########

        start.record() ###############################################
        
        cls_scores = model.bbox_head.fc_cls(query_embeds) #linear
        cls_probs = torch.softmax(cls_scores, dim=-1) #6 x 100 x 81
        bbox_preds = model.bbox_head.reg_ffn(query_embeds) #linear
        bbox_preds = model.bbox_head.activate(bbox_preds) #relu
        bbox_preds = model.bbox_head.fc_reg(bbox_preds) #linear
        bbox_preds = torch.sigmoid(bbox_preds) #6 x 100 x 4
        
        end.record()
        torch.cuda.synchronize()
        #times['output_heads'].append(start.elapsed_time(end)) ##########
        result['output_heads_time'] = start.elapsed_time(end) ##########

    
    #collect output
    # result['query_embeds'] = query_embeds.cpu().numpy()
    result['bbox_preds'] = bbox_preds.cpu().numpy()
    result['cls_probs'] = cls_probs.cpu().numpy()

    result['gt_bboxes'] = gt['bboxes']
    result['gt_labels'] = gt['labels']
    output.append(result)
    
#save to pickle file
with open(pkl_fname, 'wb') as f:
    pickle.dump(output, f)
