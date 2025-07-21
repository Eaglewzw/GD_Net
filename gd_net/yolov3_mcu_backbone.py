
from torch import nn
from tinynas.nn.networks import ProxylessNASNets
import json
import os.path as osp
import torch
from thop import profile


GLOBAL_PTH_PATH = "/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth"         #模型文件
GLOBAL_JSON_PATH = "/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.json"       #模型地址

# GLOBAL_PTH_PATH = "/home/verse/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.pth"         #模型文件
# GLOBAL_JSON_PATH = "/home/verse/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.json"       #模型地址

def _make_backbone():
    feat_dims = [24, 48, 96]
    json_file = osp.join(osp.split(osp.abspath(__file__))[0], GLOBAL_JSON_PATH)
    with open(json_file) as f:
        config = json.load(f)
    _model = ProxylessNASNets.build_from_config(config)
    ckpt = torch.load(osp.join(osp.split(osp.abspath(__file__))[0], GLOBAL_PTH_PATH))
    _model.load_state_dict(ckpt['state_dict'], strict=False)
    return _model, feat_dims

class mcunet_vww_Backbone(nn.Module):
    def x1_hook(self, module, input, output):
        self.x1 = output

    def x2_hook(self, module, input, output):
        self.x2 = output

    def x3_hook(self, module, input, output):
        self.x3 = output


    def __init__(self, weight_path=None, dummy_scale=1.0):
        super().__init__()
        self.model, feats_dim= _make_backbone()
        self.model.blocks[5].register_forward_hook(self.x1_hook)
        self.model.blocks[10].register_forward_hook(self.x2_hook)
        self.model.blocks[12].register_forward_hook(self.x3_hook)
        self.features = {}

    def get_features(self, x):
        """返回多尺度特征图字典"""
        self.features.clear()

        # 获取初始卷积层输出
        x = self.model.first_conv(x)
        self.features['stem'] = x

        # 遍历各阶段并记录特征
        stage_blocks = [
            ('stage1', 0, 4),
            ('stage2', 4, 8),
            ('stage3', 8, 12),
            ('stage4', 12, len(self.model.blocks))
        ]

        for stage_name, start_idx, end_idx in stage_blocks:
            for i, block in enumerate(self.model.blocks[start_idx:end_idx]):
                x = block(x)
                self.features[f'{stage_name}_block{i}'] = x

        return self.features

    def forward(self, x):
        x = self.model(x)
        # return [self.x1, self.x2, x]
        return {"dark3": self.x1, "dark4": self.x2, "dark5": self.x3}


if __name__ == "__main__":
    import torch
    input = torch.randn(1, 3, 160, 160)
    # model = CSPDarknet(0.33, 0.25, depthwise=True)
    model = mcunet_vww_Backbone(0.33, 0.25)

    print('========================================')
    # 获取多尺度特征
    multi_scale_features = model.get_features(input)

    # 使用示例：打印特征图尺寸
    for name, feat in multi_scale_features.items():
        print(f"{name}: {feat.shape}")

    print('========================================')
    output = model(input)
    for k in output:
        print(output[k].shape)
    # print(model)


    flops, params = profile(model, inputs=(input, ), verbose=False)
    print('========================================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))


    print('========================================')
    dtype = next(model.parameters()).dtype
    if dtype == torch.float16:
        print("模型是 FP16 (半精度)")
    elif dtype == torch.float32:
        print("模型是 FP32 (单精度)")
    else:
        print(f"未知数据类型: {dtype}")
