import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS, ATTENTION,\
        FEEDFORWARD_NETWORK, POSITIONAL_ENCODING
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg

#@BLOCKS.register_module()
@FEEDFORWARD_NETWORK.register_module()
class DETRDecoder(BaseModule):
    def __init__(self,
            num_layers=6,
            self_attn_cfg=None,
            cross_attn_cfg=None,
            ffn_cfg=None,
            out_norm_cfg=dict(type='LN'),
            return_all_layers=False,
            init_cfg=None
        ):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.self_attns = [build_from_cfg(self_attn_cfg, ATTENTION) for i in range(num_layers)]
        self.cross_attns = [build_from_cfg(cross_attn_cfg, ATTENTION) for i in range(num_layers)]
        self.ffns = [build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK) for i in range(num_layers)]
        self.dim = self.self_attns[0].attn.qk_dim
        self.return_all_layers = return_all_layers
        
        self.self_attns = nn.ModuleList(self.self_attns)
        self.cross_attns = nn.ModuleList(self.cross_attns)
        self.ffns = nn.ModuleList(self.ffns)
        self.out_norm = build_norm_layer(out_norm_cfg, self.dim)[1]
    
    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, embeds, embeds_pos, feats, feats_pos, offset=0):
        output = []
        for i in range(self.num_layers):
            embeds = self.self_attns[i](embeds, embeds_pos)
            embeds = self.cross_attns[i](embeds, feats, embeds_pos, feats_pos, offset=offset)
            embeds = self.ffns[i](embeds)
            output.append(embeds)
        output = torch.stack(output, dim=0)
        if not self.return_all_layers:
            output = output[-1]
        output = self.out_norm(output)
        return output

        

@FEEDFORWARD_NETWORK.register_module()
class DETREncoder(BaseModule):
    def __init__(self,
            num_layers=6,
            self_attn_cfg=None,
            ffn_cfg=None,
            out_norm_cfg=dict(type='LN'),
            init_cfg=None
        ):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.self_attns = [build_from_cfg(self_attn_cfg, ATTENTION) for i in range(num_layers)]
        self.self_attns = nn.ModuleList(self.self_attns)

        self.ffns = None
        if ffn_cfg is not None:
            self.ffns = [build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK) for i in range(num_layers)]
            self.ffns = nn.ModuleList(self.ffns)
        
        self.dim = self.self_attns[0].attn.qk_dim

        self.out_norm = None
        if out_norm_cfg is not None:
            self.out_norm = build_norm_layer(out_norm_cfg, self.dim)[1]
        

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, feats, feats_pos, offset=0):
        for i in range(self.num_layers):
            feats = self.self_attns[i](feats, feats_pos, offset=offset)
            if self.ffns is not None:
                feats = self.ffns[i](feats)
        if self.out_norm is not None:
            feats = self.out_norm(feats)
        return feats
