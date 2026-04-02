import tempfile
import os
import logging
from thop import clever_format, profile
import torch
from gd_net.model import YOLOv3_McuNet
from cfg import cfg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def print_model_size(model, input_size=(1, 3, 256, 256), device='cuda'):
    """
    全面打印模型信息（增强版）：
    - 总参数量 / FLOPs / 权重文件大小
    - 精细化各子模块参数量（Backbone、Neck、FPN、Heads、Pred Layers）
    """
    model.eval()
    model = model.to(device)
    x = torch.randn(input_size).to(device)

    # ====================== 辅助函数：统计模块参数 ======================
    def count_params(module, name=""):
        if module is None:
            return 0
        params = sum(p.numel() for p in module.parameters())
        print(f"  ├─ {name:<28}: {params:,} params  ({params / 1e6:.3f} M)")
        return params

    # ====================== 1. 总参数量统计 ======================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"=" * 70)
    print(f"{' MODEL SUMMARY ':^70}")
    print(f"=" * 70)
    print(f"{'Input size':<30}: {input_size}")
    print(f"{'Total params':<30}: {total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"{'Trainable params':<30}: {trainable_params:,} ({trainable_params / 1e6:.3f} M)")
    print(f"{'Non-trainable params':<30}: {non_trainable_params:,} ({non_trainable_params / 1e6:.3f} M)")

    # ====================== 2. 精细化子模块统计 ======================
    print(f"\n{' 子模块参数量详细统计 ':=^70}")

    # Backbone
    count_params(model.backbone, "Backbone (ProxylessNAS)")

    # Neck
    count_params(model.neck, "Neck (SPPF / C2f etc.)")

    # FPN
    count_params(model.fpn, "FPN / PAN")

    # 3 个 Decoupled Head
    print(f"  ├─ {'Decoupled Heads (×3)':<28}:")
    head_total = 0
    for i, head in enumerate(model.non_shared_heads):
        p = sum(p.numel() for p in head.parameters())
        head_total += p
        print(f"  │   ├─ Level {i} Head{' ':<15}: {p:,} params  ({p / 1e3:.1f} K)")
    print(f"  │   {'─' * 36}")
    print(f"  │   ├─ 3×Head Total{' ':<18}: {head_total:,} params  ({head_total / 1e6:.3f} M)")

    # 1×1 预测层（obj + cls + reg）
    print(f"  ├─ {'1×1 Prediction Layers (×3)':<28}:")
    pred_total = 0
    for i in range(len(model.obj_preds)):
        p_obj = sum(p.numel() for p in model.obj_preds[i].parameters())
        p_cls = sum(p.numel() for p in model.cls_preds[i].parameters())
        p_reg = sum(p.numel() for p in model.reg_preds[i].parameters())
        total_i = p_obj + p_cls + p_reg
        pred_total += total_i
        print(f"  │   ├─ Level {i} (obj+cls+reg){' ':<10}: {total_i:,} params")
    print(f"  │   {'─' * 36}")
    print(f"  │   ├─ 3×Pred Total{' ':<19}: {pred_total:,} params  ({pred_total / 1e3:.1f} K)")

    # ====================== 3. FLOPs（thop） ======================
    print(f"\n{' 计算量 (FLOPs) ':=^70}")
    macs, params = profile(model, inputs=(x,), verbose=False)
    flops = macs * 2
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"  ├─ FLOPs{' ':<25}: {flops_str}")
    print(f"  ├─ Params (thop){' ':<18}: {params_str}")

    # ====================== 4. 模型权重文件大小 ======================
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / 1e6
        os.unlink(f.name)
    print(f"  ├─ Model size on disk{' ':<14}: {size_mb:.2f} MB")

    print(f"=" * 70)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = YOLOv3_McuNet(
        cfg=cfg,
        device=device,
        num_classes=1,
        trainable=False,
        deploy=False
    ).to(device)

    # 你可以自由切换输入分辨率看不同大小
    print_model_size(model, input_size=(1, 3, 320, 320), device=device)  # MCU 版 64×64
    # print_model_size(model, input_size=(1, 3, 640, 640), device=device)  # 标准 640×640