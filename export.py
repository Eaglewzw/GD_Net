"""
export.py  ── 将训练好的 .pth 模型导出为 ONNX，并做基本验证

导出模式（deploy=True）：
    模型输出 [N_anchors, 4+C] 原始分数张量，由下游自行做阈值过滤和 NMS。
    这是部署到 ONNX Runtime / TFLite 的推荐模式，图结构最简单。

用法：
    python export.py --data data/drone.yaml \
                     --weights ./checkpoints/best_yolov3_mcu.pth \
                     --img-size 320
"""

import argparse
import os
import warnings

import yaml
import torch
import torch.onnx

warnings.filterwarnings("ignore")

from cfg import cfg as model_cfg
from gd_net.model import YOLOv3_McuNet


# ──────────────────────────────────────────────────────────────
#  读取 yaml
# ──────────────────────────────────────────────────────────────
def load_data_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {0: 'object'})          # {0: 'car', 1: 'truck', ...}
    num_classes = len(names)
    class_names = [names[i] for i in sorted(names)]
    return num_classes, class_names


# ──────────────────────────────────────────────────────────────
#  导出函数
# ──────────────────────────────────────────────────────────────
def export_onnx(weights, output, img_size, num_classes, device, opset=17):
    device = torch.device(device)

    # 1. 构建模型（deploy=True → 输出原始分数张量，无 NMS）
    model = YOLOv3_McuNet(
        model_cfg, device=device,
        num_classes=num_classes,
        trainable=False, deploy=True,
    ).to(device)

    # 2. 加载权重
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[export] Loaded weights: {weights}")

    # 3. 虚拟输入验证一次
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"[export] PyTorch output shape: {out.shape}  "
          f"(N_anchors={out.shape[0]}, 4+C={out.shape[1]})")

    # 4. 导出
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.onnx.export(
        model, dummy, output,
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch'},
            'output': {0: 'anchors'},
        },
        dynamo=False,
        verbose=False,
    )
    print(f"[export] ONNX saved → {output}")
    return output


# ──────────────────────────────────────────────────────────────
#  验证函数
# ──────────────────────────────────────────────────────────────
def verify_onnx(onnx_path, img_size):
    import onnx
    import onnxruntime as ort
    import numpy as np

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("[verify] ONNX model check passed")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"[verify] input={inp_name}  output={out_name}")
    print(f"[verify] providers active: {sess.get_providers()}")

    dummy_np = np.zeros((1, 3, img_size, img_size), dtype=np.float32)
    result = sess.run([out_name], {inp_name: dummy_np})[0]
    print(f"[verify] ONNX Runtime output shape: {result.shape}  dtype={result.dtype}")
    print("[verify] All good ✓")


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Export PTH → ONNX")
    parser.add_argument('--data', type=str,
                        default='./data/standford.yaml',
                        help='数据集 yaml，自动读取类别数和类别名')
    parser.add_argument('--weights', type=str,
                        default='./checkpoints/best_yolov3_mcu.pth')
    parser.add_argument('--output', type=str,
                        default='./checkpoints/yolov3_mcu.onnx')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--no-verify', action='store_true',
                        help='跳过 ONNX Runtime 验证')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    num_classes, class_names = load_data_yaml(args.data)
    print(f"[export] data yaml : {args.data}")
    print(f"[export] classes   : {num_classes}  {class_names}")

    onnx_path = export_onnx(
        weights=args.weights,
        output=args.output,
        img_size=args.img_size,
        num_classes=num_classes,
        device=args.device,
        opset=args.opset,
    )
    if not args.no_verify:
        verify_onnx(onnx_path, args.img_size)
