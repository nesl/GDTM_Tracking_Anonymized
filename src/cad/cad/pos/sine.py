import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import POSITIONAL_ENCODING, FEEDFORWARD_NETWORK
from mmcv import build_from_cfg


#x and y are [bs1,bs2,...,bk] x n
#interleave along feat dim
#output is [bs1,bs2,...,bk] x 2n
def interleave(x, y):
    z = torch.stack([x, y], dim=-1) #bs x n x 2
    z = z.flatten(-2) #bs x 2*n
    return z

#built from mmdetection and CondDETR implementations
#code is very different but computes the same thing
class SineTransform(nn.Module):
    def __init__(self, dim=128, scale=2*math.pi):
        super().__init__()
        assert dim % 2 == 0, 'embedding dim must be even'
        logspace = torch.logspace(start=1, end=-4, steps=dim//2, base=scale)
        self.register_buffer('logspace', logspace) #auto move to gpu/cpu

    #offset is [bs1,bs2,...,bk] x n 
    def forward(self, offset):
        pos = offset.unsqueeze(-1) * self.logspace #bs x dim/2
        pos = interleave(pos.sin(), pos.cos())
        return pos.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, dim=256, mode='add', out_proj=True):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.out_proj = None
        if out_proj:
            self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        pos = self.encode(x)
        if self.out_proj is not None:
            pos = self.out_proj(pos)
        return pos
        # if self.mode == 'add':
            # return x + pos
        # elif self.mode == 'cat':
            # return torch.cat((x, pos), dim=-1)
        # else:
            # print('mode %s not supported' % self.mode)
            # assert 1==2


@POSITIONAL_ENCODING.register_module()
class LearnedEncoding1d(PositionalEncoding):
    def __init__(self, num_embeds=100, dim=256, mode='add', out_proj=True):
        super().__init__(mode=mode, out_proj=out_proj)
        self.weight = nn.Embedding(num_embeds, dim).weight
        self.dim = dim

    def encode(self, x, *args, **kwargs):
        B, L, D = x.shape
        #pos = self.weight.expand(B, -1, -1)
        pos = self.weight.unsqueeze(0)
        return pos

def inverse_sigmoid(p):
    val = p / (1 - p)
    return torch.log(val + 1e-8)

@POSITIONAL_ENCODING.register_module()
class AnchorEncoding(PositionalEncoding):
    def __init__(self, dim=256, mode='add', out_proj=False, scale=2 * math.pi,
                 grid_size=(10, 10), learned=False
        ):
        super().__init__(mode=mode, out_proj=out_proj)
        self.dim = dim
        self.scale = scale
        self.sine_transform = SineTransform(dim//2, scale=scale)
        self.grid_size = grid_size
        x = torch.arange(0, grid_size[0]) / grid_size[0]
        y = torch.arange(0, grid_size[1]) / grid_size[1]
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        unscaled_params_x = inverse_sigmoid(grid_x)
        unscaled_params_y = inverse_sigmoid(grid_y)
        if learned:
            self.unscaled_params_x = nn.Parameter(unscaled_params_x)
            self.unscaled_params_y = nn.Parameter(unscaled_params_y)
        else:
            self.register_buffer('unscaled_params_x', unscaled_params_x)
            self.register_buffer('unscaled_params_y', unscaled_params_y)

    def encode(self, x=None):
        grid_x = self.unscaled_params_x.sigmoid()
        grid_y = self.unscaled_params_y.sigmoid()
        pos_x = self.sine_transform(grid_x)
        pos_y = self.sine_transform(grid_y)

        # B, L, C = x.shape
        # ref_points = self.ffn(x).sigmoid()
        # y, x = ref_points[..., 0], ref_points[..., 1]
        pos = torch.cat([pos_x, pos_y], dim=-1)
        return pos #grid_size x dim


@POSITIONAL_ENCODING.register_module()
class RefPointEncoding(PositionalEncoding):
    def __init__(self, dim=256, mode='add', out_proj=True, scale=2 * math.pi,
                 ffn_cfg=dict(type='MLP', in_channels=256, hidden_channels=[256], out_channels=2),
        ):
        super().__init__(mode=mode, out_proj=out_proj)
        self.dim = dim
        self.scale = scale
        self.sine_transform = SineTransform(dim//2, scale=scale)
        self.ffn = build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK)

    def encode(self, x):
        B, L, C = x.shape
        ref_points = self.ffn(x).sigmoid()
        y, x = ref_points[..., 0], ref_points[..., 1]
        pos_y = self.sine_transform(y)
        pos_x = self.sine_transform(x)
        pos = torch.cat([pos_y, pos_x], dim=-1)
        return pos

@POSITIONAL_ENCODING.register_module()
class SineEncoding2d(PositionalEncoding):
    def __init__(self, dim=256, mode='add', out_proj=True, scale=2 * math.pi):
        super().__init__(mode=mode, out_proj=out_proj)
        self.dim = dim
        self.scale = scale
        self.sine_transform = SineTransform(dim//2, scale=scale)

    def encode(self, x):
        B, H, W, C = x.shape
        mask = torch.ones(H, W)
        mask = mask.to(dtype=torch.int, device=x.device)
        # mask = 1 - mask  # logical_not
        y = mask.cumsum(0, dtype=torch.float32)
        x = mask.cumsum(1, dtype=torch.float32)
        pos_y = self.sine_transform(y / y.max())
        pos_x = self.sine_transform(x / x.max())
        pos = torch.cat([pos_y, pos_x], dim=-1)
        pos = pos.unsqueeze(0)
        pos = pos.expand(B, -1, -1, -1)
        return pos

    def encode_from_mask(self, mask):
        B, H, W = mask.shape
        # B, H, W, C = x.shape
        # mask = torch.ones(H, W)
        # mask = mask.to(dtype=torch.int, device=x.device)
        # mask = 1 - mask  # logical_not
        y = mask.cumsum(1, dtype=torch.float32)
        x = mask.cumsum(2, dtype=torch.float32)
        pos_y = self.sine_transform(y / y.max())
        pos_x = self.sine_transform(x / x.max())
        pos = torch.cat([pos_y, pos_x], dim=-1)
        #pos = pos.unsqueeze(0)
        # pos = pos.expand(B, -1, -1, -1)
        return pos
