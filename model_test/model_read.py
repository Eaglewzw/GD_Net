import sys
import os
import json
import os.path as osp
import torch
from torch import nn
from tinynas.nn.networks import ProxylessNASNets

# ---------------------------------------------------
# 路径配置
# ---------------------------------------------------
GLOBAL_PTH_PATH = "/home/verser/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.pth"
GLOBAL_JSON_PATH = "/home/verser/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.json"


def _build_raw_mcunet():
    json_file = GLOBAL_JSON_PATH
    if not osp.exists(json_file):
        raise FileNotFoundError(f"配置文件未找到: {json_file}")

    with open(json_file, 'r') as f:
        config = json.load(f)

    raw_model = ProxylessNASNets.build_from_config(config)

    ckpt_path = GLOBAL_PTH_PATH
    if osp.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        raw_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}")

    return raw_model, config


class mcunet_vww_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.config = _build_raw_mcunet()

        if hasattr(self.model, 'classifier'):
            del self.model.classifier
        if hasattr(self.model, 'feature_mix_layer'):
            del self.model.feature_mix_layer

        self.out_indices = {5: 0, 10: 1, 15: 2}
        self.max_idx = max(self.out_indices.keys())
        self.feat_dims = self._parse_dims()

    def _parse_dims(self):
        dims = []
        sorted_items = sorted(self.out_indices.items(), key=lambda x: x[1])
        for block_idx, _ in sorted_items:
            try:
                block_cfg = self.config['blocks'][block_idx]
                c = block_cfg['mobile_inverted_conv']['out_channels']
                dims.append(c)
            except:
                dims.append(0)
        return dims

    def forward(self, x):
        features = [None, None, None]

        # --- 打印输入 ---
        print(f"Input Tensor    | {x.shape}")

        # 1. Stem
        x = self.model.first_conv(x)
        print(f"Layer: Stem     | {x.shape}")

        # 2. Blocks
        for i, block in enumerate(self.model.blocks):
            x = block(x)

            # --- 纯净打印每一层 Block 的输出 ---
            print(f"Layer: Block {i:<2} | {x.shape}")

            if i in self.out_indices:
                mapped_index = self.out_indices[i]
                features[mapped_index] = x

            if i == self.max_idx:
                break

        return features


def mcunet_vww_make_backbone():
    backbone = mcunet_vww_Backbone()
    return backbone, backbone.feat_dims


if __name__ == "__main__":
    try:
        model, dims = mcunet_vww_make_backbone()

        # 这里使用你刚才的输入尺寸
        input_tensor = torch.randn(1, 3, 320, 320)

        print('\n=== Layer-wise Shape Debug ===')
        output = model(input_tensor)
        print('==============================\n')

    except Exception as e:
        print(f"\n❌ Error: {e}")