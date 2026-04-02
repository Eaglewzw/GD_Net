import torch
import torch.nn as nn

# ── 激活函数映射表（替代原来的 if-elif 链，扩展时只需加一行）──
_ACT_MAP = {
    'relu':  lambda: nn.ReLU(inplace=True),
    'lrelu': lambda: nn.LeakyReLU(0.1, inplace=True),
    'mish':  lambda: nn.Mish(inplace=True),
    'silu':  lambda: nn.SiLU(inplace=True),
}

# ── 归一化映射表 ──
_NORM_MAP = {
    'BN': lambda dim: nn.BatchNorm2d(dim),
    'GN': lambda dim: nn.GroupNorm(num_groups=32, num_channels=dim),
}


def get_activation(act_type):
    """根据名称返回激活层实例；act_type=None 时返回 None。"""
    if act_type is None:
        return None
    factory = _ACT_MAP.get(act_type)
    if factory is None:
        raise ValueError(f"Unknown activation: {act_type!r}. Supported: {list(_ACT_MAP)}")
    return factory()


def get_norm(norm_type, dim):
    """根据名称返回归一化层实例；norm_type=None 时返回 None。"""
    if norm_type is None:
        return None
    factory = _NORM_MAP.get(norm_type)
    if factory is None:
        raise ValueError(f"Unknown norm: {norm_type!r}. Supported: {list(_NORM_MAP)}")
    return factory(dim)


def _make_conv_block(c1, c2, k, p, s, d, g, norm_type, act_type):
    """构建单个 Conv→Norm→Act 序列，供 Conv 类内部复用。"""
    add_bias = norm_type is None
    layers = [nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=add_bias)]
    if norm_type:
        layers.append(get_norm(norm_type, c2))
    if act_type:
        layers.append(get_activation(act_type))
    return layers


# Basic conv layer
class Conv(nn.Module):
    def __init__(self,
                 c1,                   # in channels
                 c2,                   # out channels
                 k=1,                  # kernel size
                 p=0,                  # padding
                 s=1,                  # stride
                 d=1,                  # dilation
                 act_type='lrelu',     # activation
                 norm_type='BN',       # normalization
                 depthwise=False):
        super().__init__()
        if depthwise:
            # depthwise conv (groups=c1) + pointwise conv (groups=1)
            layers = _make_conv_block(c1, c1, k, p, s, d, c1,  norm_type, act_type)
            layers += _make_conv_block(c1, c2, 1, 0, 1, d,  1, norm_type, act_type)
        else:
            layers = _make_conv_block(c1, c2, k, p, s, d, 1, norm_type, act_type)
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


# BottleNeck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(inter_dim, out_dim, k=3, p=1, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


# ResBlock
class ResBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 nblocks=1,
                 act_type='silu',
                 norm_type='BN'):
        super(ResBlock, self).__init__()
        assert in_dim == out_dim
        self.m = nn.Sequential(*[
            Bottleneck(in_dim, out_dim, expand_ratio=0.5, shortcut=True,
                       norm_type=norm_type, act_type=act_type)
                       for _ in range(nblocks)
                       ])

    def forward(self, x):
        return self.m(x)


# ConvBlocks
class ConvBlocks(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.convs = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(out_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(inter_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(out_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(inter_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return self.convs(x)



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # NPU 建议：如果 avg_pool 慢，可以尝试用 1x1 conv 替代
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 空间注意力：压缩通道，提取空间特征
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result
    