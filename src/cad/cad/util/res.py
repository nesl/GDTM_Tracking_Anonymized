import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS, FEEDFORWARD_NETWORK
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
# from ..builder import BLOCKS

class Residual(nn.Module):
    def __init__(self,
            in_channels=256, module=None,
            norm_cfg=dict(type='LN'),
            dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            init_cfg=None
        ):
        #super(Residual, self).__init__(init_cfg)
        super().__init__()
        self.module = module
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)

    def forward(self, x, *args, **kwargs):
        identity = x
        x = self.norm(x)
        x = self._forward(x, *args, **kwargs)
        x = self.dropout(x)
        x = x + identity
        return x
    
    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self._forward(x)
        x = self.dropout(x)
        x = x + identity
        return x
#single layer perceptron
#@FEEDFORWARD_NETWORK.register_module()
class Residual(BaseModule):
    def __init__(self,
            in_channels=256,
            norm_cfg=dict(type='LN'),
            dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            init_cfg=None
        ):
        super(Residual, self).__init__(init_cfg)
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
    
    # def init_weights(self):
        # pass

    def forward(self, x, *args, **kwargs):
        identity = x
        x = self.norm(x)
        x = self._forward(x, *args, **kwargs)
        x = self.dropout(x)
        x = x + identity
        return x
    
    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self._forward(x)
        x = self.dropout(x)
        x = x + identity
        return x
