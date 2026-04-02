import sys
import os
import json
import os.path as osp
import torch
from torch import nn
from tinynas.nn.networks import ProxylessNASNets
from thop import profile

# ---------------------------------------------------
# 路径配置
# ---------------------------------------------------
GLOBAL_PTH_PATH = "/home/verser/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.pth"
GLOBAL_JSON_PATH = "/home/verser/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.json"


def _build_raw_mcunet():
    """
    【新增辅助函数】
    仅负责读取配置和加载权重，生成原始的 TinyNAS 模型对象。
    """
    json_file = GLOBAL_JSON_PATH
    if not osp.exists(json_file):
        raise FileNotFoundError(f"配置文件未找到: {json_file}")

    with open(json_file, 'r') as f:
        config = json.load(f)

    # 1. 构建原始模型
    raw_model = ProxylessNASNets.build_from_config(config)

    # 2. 加载权重
    ckpt_path = GLOBAL_PTH_PATH
    if osp.exists(ckpt_path):
        # map_location='cpu' 防止在加载时占用多余显存
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        raw_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}")

    return raw_model, config


class mcunet_vww_Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 获取原始模型 (调用上面的辅助函数)
        self.model, self.config = _build_raw_mcunet()

        # 2. 清理不需要的层 (分类头等)
        if hasattr(self.model, 'classifier'):
            del self.model.classifier
        if hasattr(self.model, 'feature_mix_layer'):
            del self.model.feature_mix_layer

        # 3. 定义输出层索引
        # 这些索引对应 blocks 列表中的下标
        self.out_indices = {
            6: 0,  # P3
            12: 1,  # P4
            15: 2  # P5 (Backbone 结束处, 96通道)
        }
        self.max_idx = max(self.out_indices.keys())

        # 4. 自动解析通道数
        self.feat_dims = self._parse_dims()
        print(f"✅ MCUNet Backbone 初始化完成. Output Channels: {self.feat_dims}")

    def _parse_dims(self):
        dims = []
        # 按照 output 顺序 (0, 1, 2) 对索引进行排序
        sorted_items = sorted(self.out_indices.items(), key=lambda x: x[1])

        for block_idx, _ in sorted_items:
            try:
                block_cfg = self.config['blocks'][block_idx]
                c = block_cfg['mobile_inverted_conv']['out_channels']
                dims.append(c)
            except Exception as e:
                print(f"❌ 解析通道数失败 (Block {block_idx}): {e}")
                # 回退默认值
                dims.append(96 if block_idx == 12 else 0)
        return dims

    def forward(self, x):
        features = [None, None, None]

        # 1. Stem
        x = self.model.first_conv(x)

        # 2. Blocks
        for i, block in enumerate(self.model.blocks):
            x = block(x)

            if i in self.out_indices:
                mapped_index = self.out_indices[i]
                features[mapped_index] = x

            # --- 【核心修复】: 强制截断 ---
            # 计算完第 12 层(96通道)后立即停止，
            # 绝对不执行后面可能导致 160 通道的层。
            if i == self.max_idx:
                break

        return features


def mcunet_vww_make_backbone():
    """
    【核心修复点】
    对外接口: 兼容 YOLOv3_McuNet 的调用
    返回: (BackboneWrapper实例, 通道列表)
    """
    # 旧代码: return _raw_model, dims  <-- 错误根源
    # 新代码: 实例化我们的封装类
    backbone = mcunet_vww_Backbone()

    return backbone, backbone.feat_dims


if __name__ == "__main__":
    # 测试代码
    # 模拟外部调用
    model, dims = mcunet_vww_make_backbone()

    input_tensor = torch.randn(1, 3, 192, 256)
    print('========================================')
    output = model(input_tensor)

    for i, feat in enumerate(output):
        if feat is not None:
            print(f"Output P{i + 3} Shape: {feat.shape} | Channels: {feat.shape[1]}")