import pickle
import numpy as np
from tqdm import tqdm
import json
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
label_map = [ 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 
]


json_fname = sys.argv[1]

with open(json_fname, 'r') as f:
    output = json.load(f)

results = []
for imIdx, imKey in enumerate(output.keys()):
    fname = imKey
    img_id = int(fname.split('.')[0].split('_')[-1]) #infer img_id from filename

    if len(output[imKey])==0:
        continue

    imDets = np.array(output[imKey][0])
    bboxes = imDets[:, -5:-1] 
    probs = imDets[:, :-5]

    # Probably don't need the below one. 
    non_bg_mask = (np.sum(probs[:, :-1], axis=-1) >= 0.5)
    if np.sum(non_bg_mask) == 0:
        continue

    bboxes = bboxes[non_bg_mask]
    probs = probs[non_bg_mask]
    cls_preds = np.argmax(probs, axis=-1)#[:, np.newaxis] #100, 1
    scores = np.amax(probs, axis=-1)#[:, np.newaxis] #100, 1
    #     import pdb
    #     pdb.set_trace()

    mask2 = (scores > 0.2)

    if np.sum(mask2) == 0:
        continue
    
    bboxes = bboxes[mask2]
    scores= scores[mask2]
    cls_preds = cls_preds[mask2]

    # x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] #detr outputs center_x, center_y, w, h format
    # x1, y1, x2, y2 = x-0.5*w, y-0.5*h, x+0.5*w, y+0.5*h #convert to xyxy format
    # bboxes = np.stack([x1, y1, x2, y2], axis=-1) #xyxy format
    
    # H, W, _ = sample['ori_shape']
    # scale = np.array([W, H, W, H], dtype=np.float32)
    # bboxes *= scale

    for box, score, cls_idx in zip(bboxes, scores, cls_preds):
        box = [float(b) for b in box]
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        results.append({
            'image_id': img_id,
            'category_id': label_map[int(cls_idx)],
            'bbox': [x, y, w, h],
            'score': float(score)
        })

with open(json_fname.split('.')[0] + '_coco_format.json', 'w') as f:
    f.write(json.dumps(results, indent=4))

cocoGt = COCO('data/coco/annotations/instances_val2017.json')
cocoDt = cocoGt.loadRes(json_fname.split('.')[0] + '_coco_format.json')
imgIds = sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
