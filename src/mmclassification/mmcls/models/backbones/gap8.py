import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import BaseModule, force_fp32
from ..builder import build_backbone

# class NestedDropout(torch.nn.Module):
    # def __init__(self, a=None, p=[1], train_only=True):
        # super().__init__()
        # self.a = a
        # self.p = p
        # self.train_only = train_only

    # def forward(self, x):
        # if not self.training and self.train_only:
            # return x
        # if self.a is None:
            # num_channels = x.shape[1]
            # a = np.arange(1, num_channels + 1)
            # p = np.ones(a.shape, dtype=np.float32) / num_channels
            # samples = np.random.choice(a=a, p=p, size=x.shape[0])
        # else:
            # samples = np.random.choice(a=self.a, p=self.p, size=x.shape[0])
        # mask = torch.ones(x.shape, device=x.device, dtype=x.dtype)
        # for i, j in enumerate(samples):
            # mask[i, j:] = 0.0
        # return x * mask


class NestedDropout(torch.nn.Module):
    def __init__(self, train_only=True):
        super().__init__()
        self.train_only = train_only

    def forward(self, x):
        if not self.training and self.train_only:
            return x
        
        B, C, H, W = x.shape
        num_feats = C*H*W
        x = x.reshape(B, num_feats)

        a = np.arange(1, num_feats + 1)
        p = np.ones(a.shape, dtype=np.float32) / num_feats
        samples = np.random.choice(a=a, p=p, size=B)
        
        mask = torch.ones(x.shape, device=x.device, dtype=x.dtype)
        for i, j in enumerate(samples):
            mask[i, j:] = 0.0
        x = x * mask 
        x = x.reshape(B, C, H, W)
        return x


@BACKBONES.register_module()
class GAP8Net(BaseModule):
    def __init__(self, in_channels=1, 
                 #suffix_cfg=dict(),
                 init_cfg=[dict(type='Kaiming', layer=['Conv2d']),
                           dict(type='Constant', val=1, bias=0, layer=['_BatchNorm'])]
        ):
        super(GAP8Net, self).__init__(init_cfg)
        self.init_cfg = init_cfg
        self.in_channels = in_channels
        #self.suffix = build_backbone(suffix_cfg)
        
        bias = False
        bn = dict(type='BN')
        relu6 = dict(type='ReLU6')
        self.scale1 = nn.Sequential( #output is 8 x H/2 x W/2
            ConvModule(1, 8, kernel_size=3, padding=1, stride=2, 
                groups=1, bias=bias, norm_cfg=bn, act_cfg=relu6),
            ConvModule(8, 8, kernel_size=3, padding=1, stride=1, 
                groups=8, bias=bias, norm_cfg=bn, act_cfg=relu6)
        )
        self.scale2 = nn.Sequential( #output is 16 x H/4 x W/4
            ConvModule(8, 16, kernel_size=3, padding=1, stride=2, 
                groups=8, bias=bias, norm_cfg=bn, act_cfg=relu6),
            ConvModule(16, 16, kernel_size=3, padding=1, stride=1, 
                groups=16, bias=bias, norm_cfg=bn, act_cfg=relu6)
        )
        self.scale3 = nn.Sequential( #output is 32 x H/8 x W/8
            ConvModule(16, 32, kernel_size=3, padding=1, stride=2, 
                groups=16, bias=bias, norm_cfg=bn, act_cfg=relu6),
            ConvModule(32, 32, kernel_size=3, padding=1, stride=1, 
                groups=32, bias=bias, norm_cfg=bn, act_cfg=relu6)
        )
        self.nested_dropout = NestedDropout(train_only=True)
        self.norm = nn.BatchNorm2d(32)
        
        self.upsample = nn.Sequential( #output is 32 x H/8 x W/8
            nn.Upsample(scale_factor=2),
            ConvModule(32, 16, kernel_size=3, padding=1, stride=1, 
                groups=1, bias=bias, norm_cfg=bn, act_cfg=relu6),
            
            nn.Upsample(scale_factor=2),
            ConvModule(16, 8, kernel_size=3, padding=1, stride=1, 
                groups=1, bias=bias, norm_cfg=bn, act_cfg=relu6),
            
            nn.Upsample(scale_factor=2),
            ConvModule(8, 1, kernel_size=3, padding=1, stride=1, 
                groups=1, bias=bias, norm_cfg=bn, act_cfg=relu6),
        )
    
    def forward(self, x):
        s0 = x.mean(dim=1, keepdims=True)
        s1 = self.scale1(s0)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)
        s3 = self.nested_dropout(s3)
        s3 = self.norm(s3)
        #recon = self.upsample(s3)
        #out = self.suffix(recon)
        #scales = (s0, s1, s2, s3)
        return s3
