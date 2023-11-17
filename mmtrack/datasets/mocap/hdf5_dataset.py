from abc import ABCMeta, abstractmethod
import os
import glob
import pickle
import numpy as np
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset
from mmtrack.datasets import DATASETS
import cv2
import h5py
import torch
import json
import time
import torchaudio
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import copy
import mmcv
from mmcv.runner import get_dist_info
from matplotlib.patches import Ellipse, Rectangle
from collections import defaultdict
import torch.distributions as D
from scipy.spatial import distance
from trackeval.metrics import CLEAR, HOTA, Identity
import matplotlib
from .viz import init_fig, gen_rectange, gen_ellipse, rot2angle, points_in_rec
from mmtrack.datasets import build_dataset
import torch.nn.functional as F
from tracker import TorchMultiObsKalmanFilter

font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

@DATASETS.register_module()
class HDF5Dataset(Dataset, metaclass=ABCMeta):
    CLASSES = None
    def __init__(self,
                 cacher_cfg=None,
                 pipelines={},
                 num_past_frames=0,
                 num_future_frames=0,
                 test_mode=False,
                 limit_axis=True,
                 draw_cov=True,
                 truck_w=30,
                 truck_h=15,
                 include_z=False,
                 **kwargs):
        self.truck_w = truck_w
        self.truck_h = truck_h
        self.cacher = build_dataset(cacher_cfg)
        self.fnames, self.active_keys = self.cacher.cache()
        self.max_len = 1
        self.fps = self.cacher.fps
        self.limit_axis = limit_axis
        self.draw_cov = draw_cov
        self.num_future_frames = num_future_frames
        self.num_past_frames = num_past_frames
        self.node_pos = None
        self.node_ids = None
        self.colors = ['red', 'green', 'orange', 'black', 'yellow', 'blue']
        
        self.pipelines = {}
        for mod, cfg in pipelines.items():
            self.pipelines[mod] = Compose(cfg)

        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8) #ones?
    
    def __len__(self):
        return len(self.fnames)
    
    def apply_pipelines(self, buff):
        new_buff = {}
        for key, val in buff.items():
            mod, node = key
            if mod == 'mocap':
                new_buff[key] = val
            else:
                new_buff[key] = self.pipelines[mod](val)
        return new_buff

    def read_buff(self, ind):
        with open(self.fnames[ind], 'rb') as f:
            buff = pickle.load(f)
        return buff
    
    def __getitem__(self, ind, apply_pipelines=True):
        new_buff = self.read_buff(ind)
        if apply_pipelines:
            new_buff = self.apply_pipelines(new_buff)
        
        idx_set = torch.arange(len(self))
        start_idx = max(0, ind - self.num_past_frames)
        past_idx = idx_set[start_idx:ind]

        if len(past_idx) < self.num_past_frames:
            zeros = torch.zeros(self.num_past_frames - len(past_idx)).long()
            past_idx = torch.cat([zeros, past_idx])

        end_idx = min(ind + self.num_future_frames + 1, len(self))
        future_idx = idx_set[ind + 1:end_idx]

        if len(future_idx) < self.num_future_frames:
            zeros = torch.zeros(self.num_future_frames- len(future_idx)).long()
            future_idx = torch.cat([future_idx, zeros + len(self) - 1])
        
        buffs = []
        for idx in past_idx:
            buff = self.read_buff(idx)
            buff = self.apply_pipelines(buff)
            buffs.append(buff)
        buffs.append(new_buff)

        for idx in future_idx:
            buff = self.read_buff(idx)
            buff = self.apply_pipelines(buff)
            buffs.append(buff)
        return buffs

    
    def collect_gt(self):
        all_gt_pos, all_gt_labels, all_gt_ids, all_gt_rot, all_gt_grids = [], [], [], [], []
        for i in trange(len(self)):
            data = self[i][-1] #get last frame, eval shouldnt have future
            for key, val in data.items():
                mod, node = key
                if mod == 'mocap':
                    all_gt_pos.append(val['gt_positions'])
                    all_gt_ids.append(val['gt_ids'])
                    all_gt_rot.append(val['gt_rot'])
                    all_gt_grids.append(val['gt_grids'])
        gt = {}
        gt['all_gt_pos'] = torch.stack(all_gt_pos) #num_frames x num_objs x 3
        gt['all_gt_ids'] = torch.stack(all_gt_ids)
        gt['all_gt_rot'] = torch.stack(all_gt_rot)
        gt['all_gt_grids'] = torch.stack(all_gt_grids)
        return gt

    def eval_mot(self, outputs, gt):
        all_gt_pos = gt['all_gt_pos']
        all_gt_ids = gt['all_gt_ids']
        all_gt_rot = gt['all_gt_rot']
        all_gt_grids = gt['all_gt_grids']

        res = {}
        res['num_gt_dets'] = all_gt_ids.shape[0] * all_gt_ids.shape[1]
        res['num_gt_ids'] = len(torch.unique(all_gt_ids))
        res['num_tracker_ids'] = len(torch.unique(all_gt_ids))
        res['num_timesteps'] = len(all_gt_ids)
        res['tracker_ids'] = []
        res['gt_ids'] = all_gt_ids.numpy().astype(int)
        res['similarity_scores'] = []
        res['grid_scores'] = []
        res['nll'] = []
        res['num_tracker_dets'] = 0

        from mmtrack.models.mocap.decoderv3 import calc_grid_loss 
        all_probs, all_dists = [], []
        for i in range(res['num_timesteps']):
            pred_means = outputs['track_means'][i]
            pred_covs = outputs['track_covs'][i]
            pred_ids = outputs['track_ids'][i]
            res['num_tracker_dets'] += len(pred_ids)
            res['tracker_ids'].append(pred_ids.numpy().astype(int))
            gt_pos = all_gt_pos[i]
            gt_rot = all_gt_rot[i]
            gt_grid = all_gt_grids[i]
            
            # dist = D.MultivariateNormal(pred_means.unsqueeze(0), pred_covs.unsqueeze(0))
            # loss_vals = calc_grid_loss(dist, gt_grid)
            
            dists, probs = [], []
            scores = []
            grid_scores = []
            nll = []
            for j in range(len(pred_means)):
                # dist = torch.norm(pred_mean[j][0:2] - gt_pos[:,0:2], dim=1)
                # dists.append(dist)

                dist = D.MultivariateNormal(pred_means[j], pred_covs[j])
                # dist = D.Independent(dist, 1) #Nq independent Gaussians
                # samples = dist.sample([10000])
                
                num_gt = len(gt_pos)
                for k in range(num_gt):
                    grid = gt_grid[k]
                    pos = gt_pos[k]


                    # log_probs = dist.log_prob(grid) #*1.5
                    # logsum = torch.logsumexp(log_probs.flatten(), dim=0)
                    # scores.append(logsum.exp())
                    
                    nll.append(dist.log_prob(pos))
                    samples = dist.sample([1000])
                    angle = rot2angle(gt_rot[k], return_rads=False)
                    rec, _ = gen_rectange(gt_pos[k], angle, w=30, h=15)
                    mask = points_in_rec(samples, rec)
                    scores.append(mask.mean())
                    #scores.append(np.mean(mask))

            if len(scores) == 0:
                scores = torch.empty(len(gt_pos), 0).numpy()
            else:
                scores = torch.tensor(scores).reshape(len(pred_means), -1)
                grid_scores = torch.tensor(grid_scores).reshape(len(pred_means), -1)
                grid_scores = grid_scores.numpy().T
                scores = scores.numpy().T
                nll = torch.tensor(nll).reshape(len(pred_means), -1)

            # if len(dists) != 0:
                # dists = torch.stack(dists) #num_preds x num_gt_tracks
                # dists = dists.numpy().T
                # dists[dists > self.max_len] = self.max_len
                # dists = 1 - (dists / self.max_len)
            # else:
                # dists = torch.empty(len(gt_pos), 0).numpy()
            res['similarity_scores'].append(scores)
            res['grid_scores'].append(grid_scores)
            res['nll'].append(nll)
            # all_dists.append(dists)
            # all_probs.append(probs)
        
        scores = np.stack(res['similarity_scores']).squeeze()
        # grid_scores = np.stack(res['grid_scores']).squeeze()
        nll = np.stack(res['nll']).squeeze()
        #logdir = eval_kwargs['logdir']
        #fname = f'{logdir}/res.json'
        #met=CLEAR({'THRESHOLD': 1-(0.3/self.max_len)}) 
        met = CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False})
        out = met.eval_sequence(res)
        out = {k : float(v) for k,v in out.items()}
        
        hmet = HOTA()
        hout = hmet.eval_sequence(res)
        means = {k + '_mean' : v.mean() for k, v in hout.items()}
        hout = {k: v.tolist() for k,v in hout.items()}
        out.update(hout)
        out.update(means)

        imet = Identity({'THRESHOLD': 0.5, 'PRINT_CONFIG': False})
        iout = imet.eval_sequence(res)
        iout = {k : float(v) for k,v in iout.items()}
        out.update(iout)

        out['nll_vals'] = nll.tolist()
        out['grid_scores'] = scores.tolist()

        # with open(fname, 'w') as f:
            # json.dump(out, f)
        return out

    def track_eval(self, outputs, gt):
        res = {}
        kf = TorchMultiObsKalmanFilter(dt=1, std_acc=1)
        with torch.no_grad():
            track_output = kf.forward(outputs['det_means'], outputs['det_covs'])
        track_means = track_output[0].t()
        track_covs = track_output[1].permute(2, 0, 1)
        num_views = outputs['det_means'][0].shape[-1]
        for i in range(num_views):
            means = [mu[:, i].unsqueeze(0) for mu in outputs['det_means']]
            covs = [cov[i].unsqueeze(0) for cov in outputs['det_covs']]
            track_ids = [torch.zeros(1) for _ in outputs['det_covs']]
            new_outputs = {'track_means': means, 'track_covs': covs, 'track_ids': track_ids}
            eval_res = self.eval_mot(new_outputs, gt)
            res['det_result_%d' % (i+1)] = eval_res
        means = [mu.unsqueeze(0) for mu in track_means]
        covs = [cov.unsqueeze(0) for cov in track_covs]
        new_outputs = {'track_means': means, 'track_covs': covs, 'track_ids': track_ids}
        eval_res = self.eval_mot(new_outputs, gt)
        res['track_result'] = eval_res
        vid_outputs = new_outputs
        vid_outputs['det_means'] = outputs['det_means']
        vid_outputs['det_covs'] = outputs['det_covs']
        return res, vid_outputs
    
    def grid_search(self, outputs, gt):
        res = {}
        a_range = np.linspace(1, 10, 10).tolist()
        #a_range = [0.01, 0.05, 0.1, 0.5] + a_range
        a_range = [0.05, 0.1, 0.5] + a_range
        b_range = np.linspace(0, 500, 15).tolist()

        for a in tqdm(a_range):
            for b in b_range:
                new_outputs = {}
                new_outputs['det_means'] = outputs['det_means']
                new_outputs['det_covs'] = []
                for covs in outputs['det_covs']:
                    covs = [a * S.squeeze() + b * torch.eye(2) for S in covs]
                    # covs = torch.stack(covs)
                    # covs = a * covs + b * torch.eye(2)
                    # covs = torch.split(covs, 1)
                    # covs = [S.squeeze() for S in covs]
                    new_outputs['det_covs'].append(covs)
                try:
                    vals, _ = self.track_eval(new_outputs, gt)
                except:
                    import ipdb; ipdb.set_trace() # noqa
                res['%s_%s' % (a,b)] = vals
        return res

    def calibrate_outputs(self, outputs, calib_fname, metric='nll'):
        with open(calib_fname, 'r') as f:
            data = json.load(f)
        
        min_idx = {'det_result_%d' % (i + 1): (None, 1e20) for i in range(3)}
        for a_b, res1 in data.items():
            if a_b == 'uncalibrated':
                continue
            for det_idx, res2 in res1.items():
                if det_idx == 'track_result':
                    continue
                if metric == 'nll':
                    vals = -np.array(res2['nll_vals'])
                elif metric == 'grid_score':
                    vals = -np.array(res2['grid_scores'])
                else:
                    assert 1==2
                score = np.mean(vals)
                if min_idx[det_idx][1] > score:
                    min_idx[det_idx] = (a_b, score)
        
        min_idx = {k:v for k,v in min_idx.items() if v[0] is not None}
        min_idx = {k.split('_')[-1]: v[0].split('_') for k, v in min_idx.items()}
        min_idx = {int(k)-1 : (float(v[0]), float(v[1])) for k, v in min_idx.items()}
        print(min_idx)
        calib_outputs = {'det_means': outputs['det_means'], 'det_covs': []}
        for covs in outputs['det_covs']:
            scaled_covs = []
            for idx, cov in enumerate(covs):
                a, b = min_idx[idx]
                scaled_cov = a * cov + b * torch.eye(2)
                scaled_covs.append(scaled_cov)
            calib_outputs['det_covs'].append(scaled_covs)
        return calib_outputs


    def evaluate(self, outputs, **eval_kwargs):
        gt = self.collect_gt()
        grid_res = {}
        if eval_kwargs['grid_search']:
            grid_res = self.grid_search(outputs, gt)
        logdir = eval_kwargs['logdir']
        
        res, vid_outputs = self.track_eval(outputs, gt)
        grid_res['uncalibrated'] = res
        
        if 'calib_file' in eval_kwargs.keys():
            calib_outputs = self.calibrate_outputs(outputs, eval_kwargs['calib_file'], eval_kwargs['calib_metric'])
            res, vid_outputs = self.track_eval(calib_outputs, gt)
            grid_res['calibrated'] = res
	#CHANGED
        if 'calib_file' in eval_kwargs.keys():
            absErrorFile = f'{logdir}/mean.txt'
            with open(absErrorFile, 'w') as f:
                total = 0
                for i in range(len(gt['all_gt_pos'])):
                    error = ( (gt['all_gt_pos'][i][0][0].item() - vid_outputs['track_means'][i][0][0].item()) ** 2 + 
                        (gt['all_gt_pos'][i][0][1].item() - vid_outputs['track_means'][i][0][1].item()) ** 2 ) ** 0.5
                    total += error
                    f.write(str(error) + "\n")
                f.write("The mean is " + str(total / len(gt['all_gt_pos'])) + "\n")
                f.write("The nll loss is " + str(np.median(grid_res['calibrated']['track_result']['nll_vals'])))
                f.close()	
        fname = f'{logdir}/res.json'
        with open(fname, 'w') as f:
            json.dump(grid_res, f)

        metrics = eval_kwargs['metric']
        if 'vid' in metrics:
            self.write_video(vid_outputs, **eval_kwargs)
        return res

    def write_video(self, outputs=None, **eval_kwargs): 
        logdir = eval_kwargs['logdir']
        video_length = len(self)
        if 'video_length' in eval_kwargs.keys():
            video_length = eval_kwargs['video_length']
        fname = f'{logdir}/latest_vid.mp4'
        fig, axes = init_fig(self.active_keys)
        size = (fig.get_figwidth()*50, fig.get_figheight()*50)
        size = tuple([int(s) for s in size])
        vid = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size)

        markers, colors = [], []
        for i in range(100):
            markers.append(',')
            markers.append('o')
            colors.extend(['green', 'red', 'black', 'yellow'])

        frame_count = 0
        
        id2dist = defaultdict(list)
        for i in trange(video_length):
            data = self[i][-1] #get last frame, eval shouldnt have future
            data = self.__getitem__(i, apply_pipelines=False)[-1]
            save_frame = False
            for key, val in data.items():
                mod, node = key
                if mod == 'mocap':
                    save_frame = True
                    axes[key].clear()
                    axes[key].grid('on', linewidth=3)
                    # axes[key].set_facecolor('gray')
                    if self.limit_axis:
                        axes[key].set_xlim(0,700)
                        axes[key].set_ylim(0,500)
                        axes[key].set_aspect('equal')

                    num_nodes = len(val['node_pos'])
                    
                    for j in range(num_nodes):
                        pos = val['node_pos'][j]
                        node_id = val['node_ids'][j] + 1
                        pos = pos + 250
                        axes[key].scatter(pos[0], pos[1], marker='$N%d$' % node_id, color='black', lw=1, s=20*4**2)
                    
                    num_gt = len(val['gt_positions'])
                    for j in range(num_gt):
                        pos = val['gt_positions'][j]
                        pos = pos + 250
                        if pos[0] == -1:
                            continue
                        rot = val['gt_rot'][j]
                        ID = val['gt_ids'][j]
                        grid = val['gt_grids'][j]
                        marker = markers[ID]
                        color = colors[ID]
                        
                        axes[key].scatter(pos[0], pos[1], marker=markers[ID], color=color) 
                        
                        angle = rot2angle(rot, return_rads=False)
                        rec, _ = gen_rectange(pos, angle, w=self.truck_w, h=self.truck_h, color=color)
                        axes[key].add_patch(rec)

                        r=self.truck_w/2
                        axes[key].arrow(pos[0], pos[1], r*rot[0], r*rot[1], head_width=0.05*100, head_length=0.05*100, fc=color, ec=color)
                            
                    if outputs is not None: 
                        if len(outputs['det_means']) > 0:
                            pred_means = outputs['det_means'][i].t()
                            pred_means = pred_means + 250
                            pred_covs = outputs['det_covs'][i]
                            for j in range(len(pred_means)):
                                mean = pred_means[j].cpu()
                                cov = pred_covs[j].cpu()
                                ID = str(j+1)
                                axes[key].scatter(mean[0], mean[1], color='black', marker='$%s$' % ID, lw=1, s=20*4**2)
                                ellipse = gen_ellipse(mean, cov, edgecolor='black', fc='None', lw=2, linestyle='--')
                                axes[key].add_patch(ellipse)
                        
                        # if 'track_means' in outputs.keys() and len(outputs['track_means'][i]) > 0:
                        pred_means = outputs['track_means'][i] 
                        pred_means = pred_means + 250
                        pred_covs = outputs['track_covs'][i]
                        #pred_rots = outputs['track_rot'][i]
                        ids = outputs['track_ids'][i].to(int)
                        # slot_ids = outputs['slot_ids'][i].to(int)
                        print(pred_means, pred_covs)
                        for j in range(len(pred_means)):
                            #rot = pred_rots[j]
                            #angle = torch.arctan(rot[0]/rot[1]) * 360
                            mean = pred_means[j]
                            color = self.colors[j % len(self.colors)]
                            
                            #rec, _ = gen_rectange(mean, angle, w=self.truck_w, h=self.truck_h, color=color)
                            #axes[key].add_patch(rec)


                            # axes[key].scatter(mean[0], mean[1], color=color, marker=f'+', lw=1, s=20*4**2)
                            cov = pred_covs[j]
                            ID = ids[j]
                            # sID = slot_ids[j]
                            #axes[key].text(mean[0], mean[1], s=f'T${ID}$S{sID}', fontdict={'color': color})
                            axes[key].text(mean[0], mean[1], s=f'KF', fontdict={'color': color})
                            if self.draw_cov:
                                ellipse = gen_ellipse(mean, cov, edgecolor=color, fc='None', lw=2, linestyle='--')
                                axes[key].add_patch(ellipse)
                    
                    

                if mod in ['zed_camera_left', 'realsense_camera_img', 'realsense_camera_depth']:
                    # node_num = int(node[-1])
                    # A = outputs['attn_weights'][i]
                    # A = A.permute(1,0,2) 
                    # nO, nH, L = A.shape
                    # A = A.reshape(nO, nH, 4, 35)
                    # head_dists = A.sum(dim=-1)[..., node_num-1]
                    # head_dists = F.interpolate(head_dists.unsqueeze(0).unsqueeze(0), scale_factor=60)[0][0]
                    
                    # z = torch.zeros_like(head_dists)
                    # head_dists = torch.stack([head_dists,z,z], dim=-1)

                    # head_dists = (head_dists * 255).numpy()
                    # head_dists = (head_dists - 255) * -1
                    # head_dists = head_dists.astype(np.uint8)

                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key) # code = data['zed_camera_left'][:]
                    code = data[key]
                    img = cv2.imdecode(code, 1)
                    # img = data[key]['img'].data.cpu().squeeze()
                    # mean = data[key]['img_metas'].data['img_norm_cfg']['mean']
                    # std = data[key]['img_metas'].data['img_norm_cfg']['std']
                    # img = img.permute(1, 2, 0).numpy()
                    # img = (img * std) - mean
                    # img = img.astype(np.uint8)
                    #img = np.concatenate([img, head_dists], axis=0)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[key].imshow(img)

                if 'r50' in mod:
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key) # code = data['zed_camera_left'][:]
                    feat = data[key]['img'].data#[0].cpu().squeeze()
                    feat = feat.mean(dim=0).cpu()
                    feat[feat > 1] = 1
                    feat = (feat * 255).numpy().astype(np.uint8)
                    feat = np.stack([feat]*3, axis=-1)
                    #axes[key].imshow(feat, cmap='turbo')
                    axes[key].imshow(feat)

                 
                if mod == 'zed_camera_depth':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    dmap = data[key]['img'].data[0].cpu().squeeze()
                    axes[key].imshow(dmap, cmap='turbo')#vmin=0, vmax=10000)

                if mod == 'range_doppler':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    # img = data[key]['img'].data[0].cpu().squeeze().numpy()
                    img = data[key]
                    axes[key].imshow(img, cmap='turbo', aspect='auto')

                if mod == 'azimuth_static':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    img = data[key]['img'].data[0].cpu().squeeze().numpy()
                    axes[key].imshow(img, cmap='turbo', aspect='auto')

                if mod == 'mic_waveform':
                    axes[key].clear()
                    axes[key].set_title(key)
                    axes[key].set_ylim(-0.2,1)
                    img = data[key]#['img'].data[0].cpu().squeeze().numpy()
                    max_val = img[0].max()
                    min_val = img[0].min()
                    if max_val == min_val:
                        visual_sig = np.zeros(img[0].shape)
                    else:
                        visual_sig = (img[0] - min_val) / (max_val - min_val)
                    axes[key].plot(visual_sig, color='black')

            if save_frame:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                data = cv2.resize(data, dsize=size)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                # fname = f'{logdir}/frame_{frame_count}.png'
                # cv2.imwrite(fname, data)
                frame_count += 1
                vid.write(data) 

        vid.release()
