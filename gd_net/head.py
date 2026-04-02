import torch.nn as nn

from .modules import Conv


class DecoupledHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_classes=80):
        super().__init__()
        act_type   = cfg['head_act']
        norm_type  = cfg['head_norm']
        depthwise  = cfg['head_depthwise']

        # cls_out_dim: 至少要能容纳类别数，保持轻量
        self.cls_out_dim = max(out_dim, num_classes)
        # reg 只需与 FPN 输出通道对齐，无需强制膨胀
        self.reg_out_dim = out_dim

        self.cls_feats = self._build_feat_layers(
            cfg['num_cls_head'], in_dim, self.cls_out_dim, act_type, norm_type, depthwise)
        self.reg_feats = self._build_feat_layers(
            cfg['num_reg_head'], in_dim, self.reg_out_dim, act_type, norm_type, depthwise)

    @staticmethod
    def _build_feat_layers(num_layers, in_dim, out_dim, act_type, norm_type, depthwise):
        """构建 num_layers 层 3×3 Conv 序列（第0层做通道变换，后续层保持通道数）。"""
        layers = [Conv(in_dim, out_dim, k=3, p=1, s=1,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
        for _ in range(num_layers - 1):
            layers.append(Conv(out_dim, out_dim, k=3, p=1, s=1,
                               act_type=act_type, norm_type=norm_type, depthwise=depthwise))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.cls_feats(x), self.reg_feats(x)


# build detection head
def build_head(cfg, in_dim, out_dim, num_classes=80):
    return DecoupledHead(cfg, in_dim, out_dim, num_classes)
