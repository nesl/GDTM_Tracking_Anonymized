import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS, FEEDFORWARD_NETWORK
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg

#single layer perceptron
@FEEDFORWARD_NETWORK.register_module()
class MLP(BaseModule):
    def __init__(self,
                 in_channels=256,
                 hidden_channels=[256],
                 out_channels=80,
                 act_cfg=dict(type='GELU'),
                 init_cfg=None
        ):
        super(MLP, self).__init__(init_cfg)
        old_dim = in_channels
        layers = []
        for hc in hidden_channels:
            layers.append(nn.Linear(old_dim, hc))
            layers.append(build_from_cfg(act_cfg, ACTIVATION_LAYERS))
            old_dim = hc
        layers.append(nn.Linear(old_dim, out_channels))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
