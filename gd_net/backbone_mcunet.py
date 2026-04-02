import sys
import os
import json
import os.path as osp
import torch
from torch import nn
from tinynas.nn.networks import ProxylessNASNets

# ==============================================================================
# 1. 模型仓库配置 (Model Zoo Configuration)
# ==============================================================================
# 在这里统一管理不同模型的路径和特征层索引
MODEL_ZOO = {
    'vww': {
        'name': 'mcunet-10fps_vww',
        'json': "/home/verser/Python/GD_Net/mcunet_model/mcunet-10fps_vww.json",
        'pth': "/home/verser/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth",
        # P3, P4, P5 对应的 Block 索引
        'out_indices': {
            5: 0,  # P3
            10: 1,  # P4
            12: 2  # P5 (Backbone 结束)
        }
    },
    'imagenet': {
        'name': 'mcunet-512kb-2mb_imagenet',
        'json': "/home/verser/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.json",
        'pth': "/home/verser/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.pth",
        # P3, P4, P5 对应的 Block 索引 (ImageNet 版结构更深)
        'out_indices': {
            6: 0,  # P3
            12: 1,  # P4
            15: 2  # P5 (Backbone 结束)
        }
    }
}


# ==============================================================================
# 2. 基础构建函数
# ==============================================================================
def _build_raw_mcunet(json_path, pth_path):
    """
    仅负责根据传入的路径加载 TinyNAS 原始模型
    """
    if not osp.exists(json_path):
        raise FileNotFoundError(f"❌ 配置文件未找到: {json_path}")

    # 加载配置
    with open(json_path, 'r') as f:
        config = json.load(f)

    # 构建模型结构
    raw_model = ProxylessNASNets.build_from_config(config)

    # 加载权重
    if osp.exists(pth_path):
        print(f"�� Loading weights from: {osp.basename(pth_path)} ...")
        ckpt = torch.load(pth_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # strict=False 防止一些分类头不匹配的报错
        raw_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ 警告: 未找到权重文件 {pth_path}，将使用随机初始化！")

    return raw_model, config


# ==============================================================================
# 3. 通用 Backbone 类
# ==============================================================================
class MCUNetBackbone(nn.Module):
    def __init__(self, model_type='vww'):
        super().__init__()

        # 1. 检查模型类型是否存在
        if model_type not in MODEL_ZOO:
            raise ValueError(f"Unknown model_type: '{model_type}'. Supported: {list(MODEL_ZOO.keys())}")

        self.model_info = MODEL_ZOO[model_type]
        print(f"�� Initializing MCUNet Backbone: [{model_type.upper()}]")

        # 2. 获取原始模型
        self.model, self.config = _build_raw_mcunet(
            self.model_info['json'],
            self.model_info['pth']
        )

        # 3. 清理不需要的层 (分类头等)
        if hasattr(self.model, 'classifier'):
            del self.model.classifier
        if hasattr(self.model, 'feature_mix_layer'):
            del self.model.feature_mix_layer

        # 4. 设置输出索引 (从配置字典中读取)
        self.out_indices = self.model_info['out_indices']
        self.max_idx = max(self.out_indices.keys())

        # 5. 自动解析通道数
        self.feat_dims = self._parse_dims()
        print(f"✅ Backbone Ready. Output Channels: {self.feat_dims} (Indices: {list(self.out_indices.keys())})")

    def _parse_dims(self):
        dims = []
        # 按照 output 顺序 (0, 1, 2) 对索引进行排序
        sorted_items = sorted(self.out_indices.items(), key=lambda x: x[1])

        for block_idx, _ in sorted_items:
            try:
                # 获取对应 Block 的输出通道配置
                block_cfg = self.config['blocks'][block_idx]
                c = block_cfg['mobile_inverted_conv']['out_channels']
                dims.append(c)
            except Exception as e:
                print(f"❌ 解析通道数失败 (Block {block_idx}): {e}")
                # 最后的保底逻辑
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

            # --- 强制截断 ---
            # 运行到最深的一个需要的层就停止，节省算力
            if i == self.max_idx:
                break

        return features


# ==============================================================================
# 4. 外部调用接口
# ==============================================================================
def build_mcunet_backbone(model_type='vww'):
    """
    工厂函数
    Args:
        model_type (str): 'vww' 或 'imagenet'
    Returns:
        (backbone, feat_dims)
    """
    backbone = MCUNetBackbone(model_type=model_type)
    return backbone, backbone.feat_dims


if __name__ == "__main__":
    # ================= 测试区域 =================

    # 1. 测试 VWW 模型
    print("\n---------- Testing VWW Model ----------")
    model_vww, dims_vww = build_mcunet_backbone(model_type='vww')
    input_vww = torch.randn(1, 3, 192, 192)  # VWW 常用小尺寸
    out_vww = model_vww(input_vww)
    for i, f in enumerate(out_vww):
        if f is not None: print(f"VWW P{i + 3}: {f.shape}")

    # 2. 测试 ImageNet 模型
    print("\n---------- Testing ImageNet Model ----------")
    model_img, dims_img = build_mcunet_backbone(model_type='imagenet')
    input_img = torch.randn(1, 3, 256, 256)  # ImageNet 模型可以处理大一点的图
    out_img = model_img(input_img)
    for i, f in enumerate(out_img):
        if f is not None: print(f"ImageNet P{i + 3}: {f.shape}")