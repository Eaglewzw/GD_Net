import torch
import torch.nn as nn
import time
from thop import profile
from mcunet.model_zoo import build_model

class MCUNetBackbone(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        # 1. 构建原始模型（不包含最后的分类层）
        self.base_model, self.image_size, _ = build_model(
            net_id="mcunet-512kB",
            pretrained=False
        )

        # 2. 移除原始分类器
        if hasattr(self.base_model, 'classifier'):
            del self.base_model.classifier

        # 3. 加载预训练权重（可选）
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            self.base_model.load_state_dict(state_dict, strict=False)

        # 4. 获取特征维度信息
        self.feature_channels = self._get_feature_dims()

    def _get_feature_dims(self):
        """获取各阶段输出通道数（需根据实际模型结构调整）"""
        # 示例值，实际需要根据模型结构修改
        return {
            'stage1': 16,
            'stage2': 24,
            'stage3': 40,
            'stage4': 96
        }

    def forward(self, x):
        """返回多尺度特征图"""
        features = {}

        # 假设原始模型由blocks组成（需要根据实际模型结构调整）
        x = self.base_model.first_conv(x)  # 初始卷积层
        features['stem'] = x

        # 遍历各阶段（示例结构）
        for i, block in enumerate(self.base_model.blocks[:4]):
            x = block(x)
            features[f'stage1_{i}'] = x

        for i, block in enumerate(self.base_model.blocks[4:8]):
            x = block(x)
            features[f'stage2_{i}'] = x

        for i, block in enumerate(self.base_model.blocks[8:12]):
            x = block(x)
            features[f'stage3_{i}'] = x

        for i, block in enumerate(self.base_model.blocks[12:]):
            x = block(x)
            features[f'stage4_{i}'] = x

        return features

# 使用示例
if __name__ == "__main__":
    # 初始化Backbone
    backbone = MCUNetBackbone("/home/verse/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.pth")

    # 测试输入
    dummy_input = torch.randn(1, 3, backbone.image_size, backbone.image_size)
    features = backbone(dummy_input)

    # 打印特征图尺寸
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")

    t0 = time.time()
    outputs = backbone(dummy_input)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for name, feat in outputs.items():
        print(f"{name}: {feat.shape}")
    print('==============================')
    flops, params = profile(backbone, inputs=(dummy_input, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))