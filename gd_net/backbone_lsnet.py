import torch
import torch.nn as nn
import itertools
import math
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# ==============================================================================
# 1. 基础组件
# ==============================================================================

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


# ============================ 核心修改区域 ============================

class SKA(nn.Module):
    """
    SKA Attention (Slicing Kernel Attention) - 真实实现
    """

    def __init__(self, dim, sks, groups):
        super().__init__()
        self.dim = dim
        self.sks = sks  # small kernel size (usually 3)
        self.groups = groups

        # 使用 Unfold 提取滑动窗口数据
        # padding 保证输出 H, W 不变
        self.unfold = nn.Unfold(kernel_size=sks, padding=(sks - 1) // 2)

    def forward(self, x, w):
        """
        x: Input features [B, C, H, W]
        w: Attention map from LKP [B, C//G, sks^2, H, W]
        """
        B, C, H, W = x.shape

        # 1. 展开输入 x
        # [B, C, H, W] -> [B, C*sks*sks, L] where L=H*W
        x_unfolded = self.unfold(x)

        # 2. 重塑维度以匹配 groups
        # -> [B, groups, C//groups, sks*sks, H, W]
        x_unfolded = x_unfolded.view(B, self.groups, C // self.groups, self.sks ** 2, H, W)

        # 3. 对齐权重 w
        # w 来自 LKP: [B, C//groups, sks*sks, H, W]
        # 增加 groups 维度以便广播: [B, 1, C//groups, sks*sks, H, W]
        w = w.unsqueeze(1)

        # 4. 加权聚合 (Slicing Kernel Attention)
        # x * w 进行哈达玛积，然后在 kernel 维度 (dim=3) 求和
        out = (x_unfolded * w).sum(dim=3)

        # 5. 恢复形状
        # [B, groups, C//groups, H, W] -> [B, C, H, W]
        out = out.reshape(B, C, H, W)

        return out


class LKP(nn.Module):
    """ Large Kernel Processing """

    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w


class LSConv(nn.Module):
    """ LSNet 的核心卷积模块 """

    def __init__(self, dim):
        super(LSConv, self).__init__()
        # 定义核心参数
        self.lks = 7  # large kernel size
        self.sks = 3  # small kernel size
        self.groups = 8

        # 初始化 LKP
        self.lkp = LKP(dim, lks=self.lks, sks=self.sks, groups=self.groups)

        # 【修改】初始化 SKA，传入对应参数
        self.ska = SKA(dim, sks=self.sks, groups=self.groups)

        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # LKP 生成注意力权重，SKA 进行特征聚合
        return self.bn(self.ska(x, self.lkp(x))) + x


# ============================ 核心修改结束 ============================

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class Block(torch.nn.Module):
    def __init__(self,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1):
        super().__init__()

        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            # LSConv 内部包含 SKA 和 LKP
            self.mixer = LSConv(ed)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))


# ==============================================================================
# 2. LSNet 主网络结构 (适配版)
# ==============================================================================

class LSNet(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4],
                 pretrained=False):
        super().__init__()

        # --- Stem / Patch Embed (Stride 8) ---
        self.patch_embed = torch.nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
        )

        resolution = img_size // 8

        # --- Stages ---
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        # 丢弃 blocks4 (Stage 4)

        target_stages = 3
        blocks_list = [self.blocks1, self.blocks2, self.blocks3]

        attn_ratio = [4.0] * 4

        for i in range(target_stages):
            ed = embed_dim[i]
            kd = key_dim[i]
            dpth = depth[i]
            nh = num_heads[i]
            ar = attn_ratio[i]

            for d in range(dpth):
                blocks_list[i].append(Block(ed, kd, nh, ar, resolution, stage=i, depth=d))

            if i != target_stages - 1:
                blk = blocks_list[i + 1]
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1, groups=embed_dim[i]))
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i + 1], ks=1, stride=1, pad=0))
                resolution = (resolution - 1) // 2 + 1

        self.num_features = embed_dim[2]

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        p3 = x
        x = self.blocks2(x)
        p4 = x
        x = self.blocks3(x)
        p5 = x
        return [p3, p4, p5]


# ==============================================================================
# 3. Backbone 封装类 (供 YOLO 调用)
# ==============================================================================

class LSNetBackbone(nn.Module):
    def __init__(self, version='lsnet_t', pretrained=True):
        super().__init__()

        if 'lsnet_t' in version:
            embed_dim = [64, 128, 256, 384]
            depth = [0, 2, 8, 10]
            self.feat_dims = [64, 128, 256]
        elif 'lsnet_s' in version:
            embed_dim = [96, 192, 320, 448]
            depth = [1, 2, 8, 10]
            self.feat_dims = [96, 192, 320]
        elif 'lsnet_b' in version:
            embed_dim = [128, 256, 384, 512]
            depth = [4, 6, 8, 10]
            self.feat_dims = [128, 256, 384]
        else:
            raise ValueError(f"Unknown LSNet version: {version}")

        self.model = LSNet(
            embed_dim=embed_dim,
            depth=depth,
            pretrained=pretrained
        )

        if pretrained:
            print(f"⚠️ LSNet ({version}) Pretrained weights not loaded (Placeholder).")

    def forward(self, x):
        return self.model(x)


def build_lsnet_backbone(model_type='lsnet_t', pretrained=True):
    backbone = LSNetBackbone(model_type, pretrained)
    return backbone, backbone.feat_dims


# ==============================================================================
# 4. Main 测试函数
# ==============================================================================

if __name__ == "__main__":
    print("Testing LSNetBackbone with Real SKA...")

    # 1. 实例化模型
    backbone, feats = build_lsnet_backbone('lsnet_t', pretrained=False)
    backbone.eval()

    print(f"Model created. Feature Channels: {feats}")

    # 2. 创建输入 (1, 3, 640, 640)
    input_tensor = torch.randn(1, 3, 640, 640)

    # 3. 前向推理
    outputs = backbone(input_tensor)

    # 4. 结果验证
    print(f"\nInput Shape: {input_tensor.shape}")
    print("Output Shapes:")
    for i, out in enumerate(outputs):
        stride = 8 * (2 ** i)
        print(f"  P{i + 3} (Stride {stride}): {out.shape} | Channels: {out.shape[1]}")

    assert outputs[0].shape[-1] == 640 // 8
    assert outputs[1].shape[-1] == 640 // 16
    assert outputs[2].shape[-1] == 640 // 32
    print("\n✅ LSNetBackbone Test Passed!")