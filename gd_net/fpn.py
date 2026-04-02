import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Conv, ConvBlocks


def _build_topdown(c3, c4, c5, hidden_c3, hidden_c4, hidden_c5, act_type, norm_type):
    """构建 FPN Top-Down 所需的所有子模块，FPN 和 PANet 共用。"""
    top_down_layer_1 = ConvBlocks(c5, hidden_c5, act_type=act_type, norm_type=norm_type)
    reduce_layer_1   = Conv(hidden_c5, hidden_c4, k=1, act_type=act_type, norm_type=norm_type)
    top_down_layer_2 = ConvBlocks(c4 + hidden_c4, hidden_c4, act_type=act_type, norm_type=norm_type)
    reduce_layer_2   = Conv(hidden_c4, hidden_c3, k=1, act_type=act_type, norm_type=norm_type)
    top_down_layer_3 = ConvBlocks(c3 + hidden_c3, hidden_c3, act_type=act_type, norm_type=norm_type)
    return top_down_layer_1, reduce_layer_1, top_down_layer_2, reduce_layer_2, top_down_layer_3


def _run_topdown(c3, c4, c5, td1, red1, td2, red2, td3):
    """执行 FPN Top-Down 前向计算，返回 (p3, p4, p5)。"""
    p5     = td1(c5)
    p5_up  = F.interpolate(red1(p5), scale_factor=2.0, mode='nearest')
    p4     = td2(torch.cat([c4, p5_up], dim=1))
    p4_up  = F.interpolate(red2(p4), scale_factor=2.0, mode='nearest')
    p3     = td3(torch.cat([c3, p4_up], dim=1))
    return p3, p4, p5


def _build_out_layers(out_dim, hidden_c3, hidden_c4, hidden_c5, norm_type, act_type):
    """若需要 projection，统一构建输出层。"""
    return nn.ModuleList([
        Conv(d, out_dim, k=1, norm_type=norm_type, act_type=act_type)
        for d in [hidden_c3, hidden_c4, hidden_c5]
    ])


class Yolov3FPN(nn.Module):
    def __init__(self, in_dims=[24, 48, 96], width=1.0, depth=1.0,
                 out_dim=None, act_type='silu', norm_type='BN'):
        super().__init__()
        c3, c4, c5 = in_dims
        hidden_c3, hidden_c4, hidden_c5 = int(c3*width), int(c4*width), int(c5*width)

        (self.top_down_layer_1, self.reduce_layer_1,
         self.top_down_layer_2, self.reduce_layer_2,
         self.top_down_layer_3) = _build_topdown(
            c3, c4, c5, hidden_c3, hidden_c4, hidden_c5, act_type, norm_type)

        if out_dim is not None:
            self.out_layers = _build_out_layers(out_dim, hidden_c3, hidden_c4, hidden_c5, norm_type, act_type)
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            self.out_dim = [hidden_c3, hidden_c4, hidden_c5]

    def forward(self, features):
        c3, c4, c5 = features
        p3, p4, p5 = _run_topdown(c3, c4, c5,
                                   self.top_down_layer_1, self.reduce_layer_1,
                                   self.top_down_layer_2, self.reduce_layer_2,
                                   self.top_down_layer_3)
        out_feats = [p3, p4, p5]
        if self.out_layers is not None:
            return [layer(feat) for feat, layer in zip(out_feats, self.out_layers)]
        return out_feats


class Yolov3PANet(nn.Module):
    def __init__(self, in_dims=[24, 48, 96], width=1.0, depth=1.0,
                 out_dim=None, act_type='silu', norm_type='BN'):
        super().__init__()
        c3, c4, c5 = in_dims
        hidden_c3, hidden_c4, hidden_c5 = int(c3*width), int(c4*width), int(c5*width)

        # ── FPN Top-Down（与 Yolov3FPN 共用同一构建函数）──
        (self.top_down_layer_1, self.reduce_layer_1,
         self.top_down_layer_2, self.reduce_layer_2,
         self.top_down_layer_3) = _build_topdown(
            c3, c4, c5, hidden_c3, hidden_c4, hidden_c5, act_type, norm_type)

        # ── PANet Bottom-Up ──
        self.downsample_1    = Conv(hidden_c3, hidden_c3, k=3, s=2, p=1, act_type=act_type, norm_type=norm_type)
        self.bottom_up_layer_1 = ConvBlocks(hidden_c3 + hidden_c4, hidden_c4, act_type=act_type, norm_type=norm_type)
        self.downsample_2    = Conv(hidden_c4, hidden_c4, k=3, s=2, p=1, act_type=act_type, norm_type=norm_type)
        self.bottom_up_layer_2 = ConvBlocks(hidden_c4 + hidden_c5, hidden_c5, act_type=act_type, norm_type=norm_type)

        if out_dim is not None:
            self.out_layers = _build_out_layers(out_dim, hidden_c3, hidden_c4, hidden_c5, norm_type, act_type)
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            self.out_dim = [hidden_c3, hidden_c4, hidden_c5]

    def forward(self, features):
        c3, c4, c5 = features

        # Top-Down
        f_p3, f_p4, f_p5 = _run_topdown(c3, c4, c5,
                                         self.top_down_layer_1, self.reduce_layer_1,
                                         self.top_down_layer_2, self.reduce_layer_2,
                                         self.top_down_layer_3)
        # Bottom-Up
        pan_p4 = self.bottom_up_layer_1(torch.cat([self.downsample_1(f_p3), f_p4], dim=1))
        pan_p5 = self.bottom_up_layer_2(torch.cat([self.downsample_2(pan_p4), f_p5], dim=1))

        out_feats = [f_p3, pan_p4, pan_p5]
        if self.out_layers is not None:
            return [layer(feat) for feat, layer in zip(out_feats, self.out_layers)]
        return out_feats


def build_fpn(cfg, in_dims, out_dim=None):
    common = dict(in_dims=in_dims, out_dim=out_dim,
                  width=cfg['width'], depth=cfg['depth'],
                  act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'])
    if cfg['fpn'] == 'yolov3_fpn':
        return Yolov3FPN(**common)
    elif cfg['fpn'] == 'yolov3_panet':
        return Yolov3PANet(**common)
    raise ValueError(f"Unknown fpn: {cfg['fpn']!r}")
