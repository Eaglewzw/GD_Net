import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov3_basic import Conv, ConvBlocks


# Yolov3FPN
# class Yolov3FPN(nn.Module):
#     def __init__(self,
#                  in_dims=[256, 512, 1024],
#                  width=1.0,
#                  depth=1.0,
#                  out_dim=None,
#                  act_type='silu',
#                  norm_type='BN'):
#         super(Yolov3FPN, self).__init__()
#         self.in_dims = in_dims
#         self.out_dim = out_dim
#         c3, c4, c5 = in_dims
#
#         # P5 -> P4
#         self.top_down_layer_1 = ConvBlocks(c5, int(512*width), act_type=act_type, norm_type=norm_type)
#         self.reduce_layer_1 = Conv(int(512*width), int(256*width), k=1, act_type=act_type, norm_type=norm_type)
#
#         # P4 -> P3
#         self.top_down_layer_2 = ConvBlocks(c4 + int(256*width), int(256*width), act_type=act_type, norm_type=norm_type)
#         self.reduce_layer_2 = Conv(int(256*width), int(128*width), k=1, act_type=act_type, norm_type=norm_type)
#
#         # P3
#         self.top_down_layer_3 = ConvBlocks(c3 + int(128*width), int(128*width), act_type=act_type, norm_type=norm_type)
#
#         # output proj layers
#         if out_dim is not None:
#             # output proj layers
#             self.out_layers = nn.ModuleList([
#                 Conv(in_dim, out_dim, k=1,
#                      norm_type=norm_type, act_type=act_type)
#                 for in_dim in [int(128 * width), int(256 * width), int(512 * width)]
#             ])
#             self.out_dim = [out_dim] * 3
#
#         else:
#             self.out_layers = None
#             self.out_dim = [int(128 * width), int(256 * width), int(512 * width)]
#
#
#     def forward(self, features):
#         c3, c4, c5 = features
#
#
#         # 添加详细的调试信息
#         # print(f"\n=== FPN调试信息 ===")
#         # print(f"输入特征图尺寸:")
#         # print(f"  c3: {c3.shape} (来自backbone的浅层特征)")
#         # print(f"  c4: {c4.shape} (来自backbone的中间特征)")
#         # print(f"  c5: {c5.shape} (来自backbone的深层特征)")
#
#         # p5/32
#         p5 = self.top_down_layer_1(c5)
#
#         # p4/16
#         p5_up = F.interpolate(self.reduce_layer_1(p5), scale_factor=2.0)
#         p4 = self.top_down_layer_2(torch.cat([c4, p5_up], dim=1))
#
#         # P3/8
#         p4_up = F.interpolate(self.reduce_layer_2(p4), scale_factor=2.0)
#         p3 = self.top_down_layer_3(torch.cat([c3, p4_up], dim=1))
#
#         out_feats = [p3, p4, p5]
#
#         # output proj layers
#         if self.out_layers is not None:
#             # output proj layers
#             out_feats_proj = []
#             for feat, layer in zip(out_feats, self.out_layers):
#                 out_feats_proj.append(layer(feat))
#             return out_feats_proj
#
#         return out_feats




class Yolov3FPN(nn.Module):
    def __init__(self,
                 in_dims=[24, 48, 96],  # 默认值适配 MCUNet
                 width=1.0,
                 depth=1.0,
                 out_dim=None,
                 act_type='silu',
                 norm_type='BN'):
        super(Yolov3FPN, self).__init__()
        self.in_dims = in_dims
        c3, c4, c5 = in_dims

        # -----------------------------------------------------------------
        # 核心修改：重新定义内部通道数，适配轻量级网络
        # 原版 YOLOv3: [256, 512, 1024] -> FPN 内部用 512, 256
        # MCUNet:      [24, 48, 96]     -> 建议 FPN 内部用 96, 48 (或者更小)
        # -----------------------------------------------------------------

        # 策略：保持通道数不膨胀，或者轻微压缩。
        # 这里我们设定 FPN 的 hidden channels 等于输入的 channels
        # hidden_c5 = 96 (对应原版的512)
        # hidden_c4 = 48 (对应原版的256)
        # hidden_c3 = 24 (对应原版的128)

        hidden_c5 = int(c5 * width)
        hidden_c4 = int(c4 * width)
        hidden_c3 = int(c3 * width)

        # --- P5 处理 ---
        # input: c5 (96) -> out: hidden_c5 (96)
        self.top_down_layer_1 = ConvBlocks(c5, hidden_c5, act_type=act_type, norm_type=norm_type)
        # out: hidden_c4 (48)
        self.reduce_layer_1 = Conv(hidden_c5, hidden_c4, k=1, act_type=act_type, norm_type=norm_type)

        # --- P4 处理 ---
        # cat input: c4 (48) + p5_up (48) = 96
        # out: hidden_c4 (48)
        self.top_down_layer_2 = ConvBlocks(c4 + hidden_c4, hidden_c4, act_type=act_type, norm_type=norm_type)
        # out: hidden_c3 (24)
        self.reduce_layer_2 = Conv(hidden_c4, hidden_c3, k=1, act_type=act_type, norm_type=norm_type)

        # --- P3 处理 ---
        # cat input: c3 (24) + p4_up (24) = 48
        # out: hidden_c3 (24)
        self.top_down_layer_3 = ConvBlocks(c3 + hidden_c3, hidden_c3, act_type=act_type, norm_type=norm_type)

        # --- 输出层 ---
        if out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1, norm_type=norm_type, act_type=act_type)
                for in_dim in [hidden_c3, hidden_c4, hidden_c5]
            ])
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            # 记录最终输出通道数供 Head 使用
            self.out_dim = [hidden_c3, hidden_c4, hidden_c5]

    def forward(self, features):
        c3, c4, c5 = features

        # P5 processing
        p5 = self.top_down_layer_1(c5)

        # P5 -> P4
        p5_up = F.interpolate(self.reduce_layer_1(p5), scale_factor=2.0)
        p4_cat = torch.cat([c4, p5_up], dim=1)
        p4 = self.top_down_layer_2(p4_cat)

        # P4 -> P3
        p4_up = F.interpolate(self.reduce_layer_2(p4), scale_factor=2.0)
        p3_cat = torch.cat([c3, p4_up], dim=1)
        p3 = self.top_down_layer_3(p3_cat)

        out_feats = [p3, p4, p5]

        # Output projection (如果需要)
        if self.out_layers is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim=None):
    model = cfg['fpn']
    # build neck
    if model == 'yolov3_fpn':
        fpn_net = Yolov3FPN(in_dims=in_dims,
                            out_dim=out_dim,
                            width=cfg['width'],
                            depth=cfg['depth'],
                            act_type=cfg['fpn_act'],
                            norm_type=cfg['fpn_norm']
                            )

    return fpn_net
