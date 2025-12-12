import torch
import torch.nn as nn
import math

__all__ = ['mnv4_conv_small', 'mnv4_conv_medium']

def make_divisible(v, divisor=8, min_value=None):
    """
    确保通道数是8的倍数，对硬件加速更友好
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UniversalInvertedBottleneck(nn.Module):
    """
    MobileNetV4 核心模块 UIB
    包含 StartDW 和 MiddleDW 两个可选的 Depthwise 卷积
    """
    def __init__(self, in_c, out_c, expand_ratio, stride, kernel_size=3):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_c * expand_ratio)
        
        # 1. Start Depthwise Conv (Optional: 只有部分层有，这里简化为标准倒残差)
        # MNV4 论文中有些 block 在 expand 前有 DW，但 Conv 版本主要是标准的 Inverted Bottleneck
        # 为了适配 NPU，我们这里实现标准的 Inverted Bottleneck 结构 (Expand -> DW -> Project)
        # 这也是 MNV4-Conv-Medium 的主要构成
        
        layers = []
        # Expand
        if expand_ratio != 1:
            layers.append(ConvBN(in_c, hidden_dim, kernel_size=1))
        
        # Depthwise
        layers.append(ConvBN(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim))
        
        # Project (Linear, no activation)
        layers.append(ConvBN(hidden_dim, out_c, kernel_size=1, act=False))
        
        self.block = nn.Sequential(*layers)
        self.use_res_connect = self.stride == 1 and in_c == out_c

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)

class MobileNetV4Backbone(nn.Module):
    def __init__(self, model_size='medium'):
        super().__init__()
        
        # 定义网络配置 (根据官方 MNV4-Conv 规格)
        # 格式: [kernel, stride, out_channels, expand_ratio, num_blocks]
        if model_size == 'small':
            # C3(s8)=64, C4(s16)=96, C5(s32)=128
            self.channels = [64, 96, 128] 
            configs = [
                # Stage 0 (Stem) - s2
                [3, 2, 32, 1, 1], 
                # Stage 1 - s4
                [3, 2, 32, 1, 1], 
                [1, 1, 32, 1, 1],
                # Stage 2 - s8 (Output C3)
                [3, 2, 96, 1, 1],
                [1, 1, 64, 1, 1],
                # Stage 3 - s16 (Output C4)
                [5, 2, 96, 3, 1], # UIB start
                [3, 1, 96, 2, 1],
                [3, 1, 96, 2, 1],
                [3, 1, 96, 2, 1],
                [3, 1, 96, 2, 1], # MNV4-Small 这里比较深
                # Stage 4 - s32 (Output C5)
                [5, 2, 128, 6, 1],
                [5, 1, 128, 4, 1],
                [5, 1, 128, 4, 1],
                [5, 1, 128, 4, 1]
            ]
        elif model_size == 'medium':
            # C3(s8)=80, C4(s16)=160, C5(s32)=256 (推荐用于检测)
            self.channels = [80, 160, 256]
            configs = [
                # Stage 0
                [3, 2, 32, 1, 1],
                # Stage 1
                [3, 2, 48, 1, 1],
                [1, 1, 48, 1, 1],
                # Stage 2 (Output C3) - s8
                [3, 2, 80, 1, 1],
                [1, 1, 80, 1, 1],
                # Stage 3 (Output C4) - s16
                [5, 2, 160, 4, 1],
                [3, 1, 160, 2, 1],
                [3, 1, 160, 2, 1],
                [3, 1, 160, 2, 1],
                [3, 1, 160, 2, 1],
                [3, 1, 160, 2, 1],
                # Stage 4 (Output C5) - s32
                [5, 2, 256, 6, 1],
                [5, 1, 256, 4, 1],
                [5, 1, 256, 4, 1],
                [5, 1, 256, 4, 1],
                [5, 1, 256, 4, 1]
            ]
        
        self.layers = nn.ModuleList()
        input_channel = 32
        
        # 构建 Stem (第一层)
        self.stem = ConvBN(3, input_channel, kernel_size=3, stride=2)
        
        # 构建中间层
        current_stride = 2
        
        # 我们需要知道每一层属于哪个 stage 从而决定在哪里输出
        # Stage2(s8) -> idx 0, Stage3(s16) -> idx 1, Stage4(s32) -> idx 2
        self.out_indices = [] 
        
        layer_idx = 0
        for k, s, out_c, exp, blocks in configs:
            # 判断是否是新的一层下采样 (s=2)
            if s == 2 and input_channel != 32: # 忽略 stem 之后的第一次 s2
                current_stride *= 2
            
            for i in range(blocks):
                stride = s if i == 0 else 1
                # 使用 UIB 或 ConvBN
                # 简单起见，如果 exp=1 且 k=3/1，我们视为普通 ConvBN (MNV4 结构特性)
                if exp == 1:
                    self.layers.append(ConvBN(input_channel, out_c, kernel_size=k, stride=stride))
                else:
                    self.layers.append(UniversalInvertedBottleneck(input_channel, out_c, exp, stride, k))
                
                input_channel = out_c
                
                # 记录输出节点索引
                # 我们需要在 stride=8, 16, 32 的最后一个 block 处截取
                # 逻辑：我们在 forward 里面动态判定 stride 比较复杂，
                # 所以我们简单地按网络结构硬编码输出点
                layer_idx += 1

        # 将 ModuleList 整理为三个 Stage 以便 Forward 调用
        # 注意：这里需要根据 configs 列表手动拆分，或者在 forward 里判断
        # 为了通用性，我们采用在 Forward 中基于 feature map 尺寸判断（虽然慢一点点但通用）
        # 或者更高效的方法：手动分组
        
        if model_size == 'small':
            self.stage1 = nn.Sequential(*self.layers[0:3])   # s4
            self.stage2 = nn.Sequential(*self.layers[3:5])   # s8  -> Out1
            self.stage3 = nn.Sequential(*self.layers[5:10])  # s16 -> Out2
            self.stage4 = nn.Sequential(*self.layers[10:])   # s32 -> Out3
        elif model_size == 'medium':
            self.stage1 = nn.Sequential(*self.layers[0:3])   # s4
            self.stage2 = nn.Sequential(*self.layers[3:5])   # s8  -> Out1
            self.stage3 = nn.Sequential(*self.layers[5:11])  # s16 -> Out2
            self.stage4 = nn.Sequential(*self.layers[11:])   # s32 -> Out3

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)     # s2
        x = self.stage1(x)   # s4
        
        c3 = self.stage2(x)  # s8
        c4 = self.stage3(c3) # s16
        c5 = self.stage4(c4) # s32
        
        return [c3, c4, c5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

def mnv4_conv_small(pretrained=False):
    model = MobileNetV4Backbone(model_size='small')
    # 如果有预训练权重，在这里加载
    return model

def mnv4_conv_medium(pretrained=False):
    model = MobileNetV4Backbone(model_size='medium')
    # 如果有预训练权重，在这里加载
    return model

if __name__ == "__main__":
    # 测试代码
    model = mnv4_conv_medium()
    dummy = torch.randn(1, 3, 320, 320)
    feats = model(dummy)
    print(f"Input: {dummy.shape}")
    for i, f in enumerate(feats):
        print(f"Output {i} (Stride {8*(2**i)}): {f.shape}")