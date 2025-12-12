import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PATH = "/home/verser/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth"

class ProxylessConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, p=1, s=1, d=1,
                 groups=1, act_type='relu', norm_type='BN'):


        super(ProxylessConv, self).__init__()

        # 计算实际填充，确保输出尺寸正确
        padding = p
        if isinstance(k, int):
            padding = (k - 1) // 2 if p == 1 else p

        # 卷积层
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=k,
            stride=s, padding=padding, dilation=d, groups=groups, bias=False
        )

        # 归一化层
        if norm_type == 'BN':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'GN':
            self.norm = nn.GroupNorm(max(1, out_channels // 8), out_channels)
        else:
            self.norm = nn.Identity()

        # 激活函数
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        elif act_type == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# --------------------- MBInvertedConvLayer -----------------------
class MBInvertedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 expand_ratio=1, mid_channels=None, act_func='relu6', use_se=False):
        super(MBInvertedConvLayer, self).__init__()

        # Calculate mid channels
        if mid_channels is None:
            mid_channels = in_channels * expand_ratio

        self.use_res_connect = stride == 1 and in_channels == out_channels

        # Pointwise expansion
        self.expand_conv = None
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                ProxylessConv(in_channels, mid_channels, k=1,
                     act_type=act_func, norm_type='BN'),
            )

        # Depthwise convolution - 关键修改：使用groups=mid_channels实现深度卷积
        # 计算正确的填充，确保输出尺寸正确
        if isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2
        else:
            padding = (kernel_size[0] - 1) // 2

        self.depthwise_conv = nn.Sequential(
            ProxylessConv(mid_channels, mid_channels, k=kernel_size, p=padding, s=stride,
                 groups=mid_channels, act_type=act_func, norm_type='BN'),
        )

        # Pointwise projection
        self.project_conv = nn.Sequential(
            ProxylessConv(mid_channels, out_channels, k=1,
                 act_type=None, norm_type='BN'),  # No activation in projection
        )

    def forward(self, x):
        identity = x

        if self.expand_conv is not None:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)
        x = self.project_conv(x)

        # 只有在步长为1且通道数相同时才使用残差连接
        if self.use_res_connect:
            # 确保尺寸匹配
            if x.shape[2:] != identity.shape[2:]:
                # 如果尺寸不匹配，使用自适应池化调整identity的尺寸
                identity = F.adaptive_avg_pool2d(identity, x.shape[2:])
            x = x + identity

        return x


# --------------------- IdentityLayer -----------------------
class IdentityLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, act_func=None,
                 dropout_rate=0, ops_order="weight_bn_act"):
        super(IdentityLayer, self).__init__()
        # Identity layer just passes through the input
        if in_channels != out_channels:
            self.identity_conv = Conv(in_channels, out_channels, k=1,
                                      act_type=act_func, norm_type='BN' if use_bn else None)
        else:
            self.identity_conv = None

    def forward(self, x):
        if self.identity_conv is not None:
            return self.identity_conv(x)
        return x


# --------------------- MobileInvertedResidualBlock -----------------------
class MobileInvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 expand_ratio=1, mid_channels=None, act_func='relu6', use_se=False,
                 shortcut=None):
        super(MobileInvertedResidualBlock, self).__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mobile_inverted_conv = MBInvertedConvLayer(
            in_channels, out_channels, kernel_size, stride,
            expand_ratio, mid_channels, act_func, use_se
        )

        # Shortcut connection
        self.shortcut = None
        if shortcut is not None:
            shortcut_in_channels = shortcut['in_channels'][0] if isinstance(shortcut['in_channels'], list) else shortcut['in_channels']
            shortcut_out_channels = shortcut['out_channels'][0] if isinstance(shortcut['out_channels'], list) else shortcut['out_channels']

            self.shortcut = IdentityLayer(
                shortcut_in_channels, shortcut_out_channels,
                shortcut['use_bn'], shortcut['act_func'],
                shortcut['dropout_rate'], shortcut['ops_order']
            )

    def forward(self, x):
        out = self.mobile_inverted_conv(x)

        if self.shortcut is not None:
            identity = self.shortcut(x)
            # 确保尺寸匹配
            if out.shape[2:] != identity.shape[2:]:
                # 如果尺寸不匹配，使用自适应池化调整identity的尺寸
                identity = F.adaptive_avg_pool2d(identity, out.shape[2:])
            out = out + identity

        return out


# --------------------- ProxylessNASNets -----------------------
class ProxylessNASNets(nn.Module):
    def __init__(self,  act_type='relu6', norm_type='BN', resolution=64, num_classes=1):
        super(ProxylessNASNets, self).__init__()

        # 特征维度为 [24, 48, 96]
        self.feat_dims = [24, 48, 96]

        # First convolution layer
        self.first_conv = nn.Sequential(
            ProxylessConv(3, 16, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
        )

        # Build blocks from JSON configuration
        self.blocks = nn.ModuleList()

        # Block 0
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=16, out_channels=8, kernel_size=3, stride=1,
            expand_ratio=1, mid_channels=None, act_func=act_type, use_se=False,
            shortcut=None
        ))

        # Block 1
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=8, out_channels=16, kernel_size=5, stride=2,
            expand_ratio=6, mid_channels=48, act_func=act_type, use_se=False,
            shortcut=None
        ))

        # Block 2
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=16, out_channels=16, kernel_size=3, stride=1,
            expand_ratio=4, mid_channels=64, act_func=act_type, use_se=False,
            shortcut={'name': 'IdentityLayer', 'in_channels': [16], 'out_channels': [16],
                      'use_bn': False, 'act_func': None, 'dropout_rate': 0, 'ops_order': 'weight_bn_act'}
        ))

        # Block 3 - First feature output (24 channels)
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=16, out_channels=24, kernel_size=3, stride=2,
            expand_ratio=5, mid_channels=80, act_func=act_type, use_se=False,
            shortcut=None
        ))

        # Block 4
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=24, out_channels=24, kernel_size=3, stride=1,
            expand_ratio=4, mid_channels=96, act_func=act_type, use_se=False,
            shortcut={'name': 'IdentityLayer', 'in_channels': [24], 'out_channels': [24],
                      'use_bn': False, 'act_func': None, 'dropout_rate': 0, 'ops_order': 'weight_bn_act'}
        ))

        # Block 5
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=24, out_channels=24, kernel_size=3, stride=1,
            expand_ratio=4, mid_channels=96, act_func=act_type, use_se=False,
            shortcut={'name': 'IdentityLayer', 'in_channels': [24], 'out_channels': [24],
                      'use_bn': False, 'act_func': None, 'dropout_rate': 0, 'ops_order': 'weight_bn_act'}
        ))

        # Block 6 - Second feature output (修改为48通道)
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=24, out_channels=48, kernel_size=3, stride=2,  # 修改为48通道
            expand_ratio=5, mid_channels=120, act_func=act_type, use_se=False,
            shortcut=None
        ))

        # Block 7
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=48, out_channels=48, kernel_size=3, stride=1,  # 修改输入输出为48通道
            expand_ratio=4, mid_channels=192, act_func=act_type, use_se=False,  # 调整中间通道数
            shortcut={'name': 'IdentityLayer', 'in_channels': [48], 'out_channels': [48],
                      'use_bn': False, 'act_func': None, 'dropout_rate': 0, 'ops_order': 'weight_bn_act'}
        ))

        # Block 8
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=48, out_channels=48, kernel_size=7, stride=1,  # 修改输入输出为48通道
            expand_ratio=3, mid_channels=144, act_func=act_type, use_se=False,  # 调整中间通道数
            shortcut={'name': 'IdentityLayer', 'in_channels': [48], 'out_channels': [48],
                      'use_bn': False, 'act_func': None, 'dropout_rate': 0, 'ops_order': 'weight_bn_act'}
        ))

        # Block 9
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=48, out_channels=48, kernel_size=3, stride=1,  # 修改输入输出为48通道
            expand_ratio=3, mid_channels=144, act_func=act_type, use_se=False,  # 调整中间通道数
            shortcut=None
        ))

        # Block 10
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=48, out_channels=48, kernel_size=3, stride=1,  # 修改输入输出为48通道
            expand_ratio=4, mid_channels=192, act_func=act_type, use_se=False,  # 调整中间通道数
            shortcut={'name': 'IdentityLayer', 'in_channels': [48], 'out_channels': [48],
                      'use_bn': False, 'act_func': None, 'dropout_rate': 0, 'ops_order': 'weight_bn_act'}
        ))

        # Block 11
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=48, out_channels=96, kernel_size=7, stride=2,  # 修改输入为48，输出为96
            expand_ratio=5, mid_channels=240, act_func=act_type, use_se=False,
            shortcut=None
        ))

        # Block 12
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=96, out_channels=96, kernel_size=5, stride=1,
            expand_ratio=5, mid_channels=480, act_func=act_type, use_se=False,
            shortcut={'name': 'IdentityLayer', 'in_channels': [96], 'out_channels': [96],
                      'use_bn': False, 'act_func': None, 'dropout_rate': 0, 'ops_order': 'weight_bn_act'}
        ))

        # Block 13 - Third feature output (修改为96通道)
        self.blocks.append(MobileInvertedResidualBlock(
            in_channels=96, out_channels=96, kernel_size=7, stride=1,  # 修改输出为96通道
            expand_ratio=4, mid_channels=384, act_func=act_type, use_se=False,
            shortcut=None
        ))

        # Classifier (optional, for classification tasks)
        self.classifier = nn.Linear(96, num_classes) if num_classes > 0 else None  # 修改为96


    def forward(self, x):
        # First convolution
        x = self.first_conv(x)

        # Store feature maps at different scales
        features = []

        # Process through all blocks
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Collect multi-scale features (similar to DarkNet's c3, c4, c5)
            if i == 3:   # After block 3: 24 channels
                feat1 = x
            elif i == 6: # After block 6: 48 channels
                feat2 = x
            elif i == 13: # After block 13: 96 channels
                feat3 = x

        # print("proxyless_feat1: {}, proxyless_feat2: {}, proxyless_feat3: {}".format(
        #     feat1, feat2, feat3))
        # print("proxyless_feat1 shape: {}, proxyless_feat2 shape: {}, proxyless_feat3 shape: {}".format(
        #     feat1.shape, feat2.shape, feat3.shape))

        outputs = [feat1, feat2, feat3]

        # If classification is needed
        if self.classifier is not None:
            # Global average pooling
            x = x.mean([2, 3])  # AdaptiveAvgPool2d alternative
            x = self.classifier(x)
            return x, outputs
        else:
            return outputs


# --------------------- Functions -----------------------
def build_proxyless_backbone(pretrained=False, pretrained_path=None, resolution=160, num_classes=0):
    """Constructs a ProxylessNASNets model.
    Args:
        pretrained (bool): If True, loads pre-trained weights
        pretrained_path (str): Path to the pre-trained weights file
        resolution (int): Input image resolution
        num_classes (int): Number of output classes (0 for backbone only)
    """
    backbone = ProxylessNASNets(act_type='relu6', norm_type='BN',
                                resolution=resolution, num_classes=num_classes)
    feat_dims = backbone.feat_dims


    # Note: Pre-trained weights would need to be sourced separately
    if pretrained:
        if pretrained_path is None:
            print('Error: pretrained_path must be provided when pretrained=True')
            return backbone, feat_dims

        try:
            print(f'Loading pretrained weights from: {pretrained_path}')

            # 加载预训练权重
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # 检查checkpoint的类型
            if isinstance(checkpoint, dict):
                # 如果checkpoint是字典，可能有不同的键名
                if 'model' in checkpoint:
                    # 通常的键名是 'model'
                    checkpoint_state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # 或者 'state_dict'
                    checkpoint_state_dict = checkpoint['state_dict']
                else:
                    # 或者整个字典就是state_dict
                    checkpoint_state_dict = checkpoint
            else:
                # 如果checkpoint直接是state_dict
                checkpoint_state_dict = checkpoint

            # 获取当前模型的state_dict
            model_state_dict = backbone.state_dict()

            # 用于统计加载了多少权重
            loaded_count = 0
            total_count = len(model_state_dict.keys())

            # 匹配并加载权重
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)

                    if shape_model == shape_checkpoint:
                        model_state_dict[k] = checkpoint_state_dict[k]
                        loaded_count += 1
                    else:
                        print(f'Shape mismatch for {k}: model {shape_model} vs checkpoint {shape_checkpoint}')
                else:
                    print(f'Unused key in checkpoint: {k}')

            # 加载匹配的权重
            backbone.load_state_dict(model_state_dict, strict=False)
            print(f'Successfully loaded {loaded_count}/{total_count} parameters from pretrained weights')

        except Exception as e:
            print(f'Error loading pretrained weights: {e}')
            print('Continuing with randomly initialized weights...')

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile

    # Test the model
    model, feats = build_proxyless_backbone(pretrained=False, resolution=64, num_classes=1)
    # model, feats = build_proxyless_backbone(pretrained=True, pretrained_path=MODEL_PATH, resolution=64, num_classes=0)
    x = torch.randn(1, 3, 640, 640)

    print("Model Feature Dimensions:", feats)

    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    # for i, out in enumerate(outputs):
    #     print(f'Feature {i+1} shape: {out.shape}')

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))