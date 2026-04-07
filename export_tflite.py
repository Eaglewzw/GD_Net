"""
export_tflite.py  ── PTH → ONNX → TF SavedModel → TFLite

路线：
    pth ──export.py──> .onnx ──onnx2tf──> saved_model/ ──tflite_convert──> .tflite

支持量化：
    --quant none   : FP32（默认）
    --quant fp16   : FP16，体积减半，精度几乎不变
    --quant int8   : INT8，需要提供校准图片目录（--calib-dir），速度最快

用法：
    # FP32
    CUDA_VISIBLE_DEVICES="" python export_tflite.py --data data/standford.yaml

    # FP16
    CUDA_VISIBLE_DEVICES="" python export_tflite.py --data data/standford.yaml --quant fp16

    # INT8（需要校准集）
    CUDA_VISIBLE_DEVICES="" python export_tflite.py --data data/standford.yaml \
        --quant int8 --calib-dir /path/to/images
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import yaml


# ──────────────────────────────────────────────────────────────
#  读取 yaml
# ──────────────────────────────────────────────────────────────
def load_data_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {0: 'object'})
    num_classes = len(names)
    class_names = [names[i] for i in sorted(names)]
    return num_classes, class_names


# ──────────────────────────────────────────────────────────────
#  Step 1: ONNX → TF SavedModel（via onnx2tf）
# ──────────────────────────────────────────────────────────────
def onnx_to_saved_model(onnx_path, saved_model_dir):
    print(f"\n[Step 1] ONNX → TF SavedModel")
    print(f"  input : {onnx_path}")
    print(f"  output: {saved_model_dir}")

    cmd = [
        sys.executable, '-m', 'onnx2tf',
        '-i', onnx_path,
        '-o', saved_model_dir,
        '--non_verbose',
    ]
    result = subprocess.run(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    if result.returncode != 0:
        raise RuntimeError("onnx2tf 转换失败，请检查上方错误信息。")
    print("[Step 1] Done ✓")
    return saved_model_dir


# ──────────────────────────────────────────────────────────────
#  Step 2: TF SavedModel → TFLite
# ──────────────────────────────────────────────────────────────
def saved_model_to_tflite(saved_model_dir, tflite_path, quant, calib_dir, img_size):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf

    print(f"\n[Step 2] TF SavedModel → TFLite  (quant={quant})")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quant == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("  量化模式: FP16")

    elif quant == 'int8':
        if not calib_dir or not os.path.isdir(calib_dir):
            raise ValueError("INT8 量化需要提供有效的 --calib-dir 图片目录")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

        # 生成校准数据集
        import cv2
        img_files = [
            os.path.join(calib_dir, f)
            for f in os.listdir(calib_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:8000]  # 最多取 200 张

        def representative_dataset():
            for path in img_files:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                # TFLite 期望 NHWC
                img = img[np.newaxis]
                yield [img]

        converter.representative_dataset = representative_dataset
        print(f"  量化模式: INT8，校准图片数: {len(img_files)}")

    else:
        print("  量化模式: FP32（无量化）")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(tflite_path)), exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / 1e6
    print(f"[Step 2] Done ✓  →  {tflite_path}  ({size_mb:.2f} MB)")
    return tflite_path


# ──────────────────────────────────────────────────────────────
#  Step 3: 验证 TFLite 推理
# ──────────────────────────────────────────────────────────────
def verify_tflite(tflite_path, img_size):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf

    print(f"\n[Step 3] Verify TFLite")
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    print(f"  input : {inp_detail['name']}  shape={inp_detail['shape']}  dtype={inp_detail['dtype']}")
    print(f"  output: {out_detail['name']}  shape={out_detail['shape']}  dtype={out_detail['dtype']}")

    # 构造虚拟输入
    dtype = inp_detail['dtype']
    if dtype == np.float32:
        dummy = np.zeros(inp_detail['shape'], dtype=np.float32)
    else:
        dummy = np.zeros(inp_detail['shape'], dtype=np.int8)

    interp.set_tensor(inp_detail['index'], dummy)
    interp.invoke()
    out = interp.get_tensor(out_detail['index'])
    print(f"  output shape: {out.shape}  dtype={out.dtype}")
    print("[Step 3] All good ✓")


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Export ONNX → TFLite")
    p.add_argument('--data', type=str,
                   default='./data/standford.yaml',
                   help='数据集 yaml，读取类别信息')
    p.add_argument('--onnx', type=str,
                   default='./checkpoints/yolov3_mcu.onnx',
                   help='输入 ONNX 文件')
    p.add_argument('--saved-model-dir', type=str,
                   default='./checkpoints/saved_model',
                   help='中间 TF SavedModel 目录')
    p.add_argument('--output', type=str,
                   default='./checkpoints/yolov3_mcu.tflite',
                   help='输出 TFLite 文件')
    p.add_argument('--img-size', type=int, default=320)
    p.add_argument('--quant', type=str, default='int8',
                   choices=['none', 'fp16', 'int8'],
                   help='量化模式')
    p.add_argument('--calib-dir', type=str, default='/media/verser/robot/Dataset/standford_car/JPEGImages',
                   help='INT8 校准图片目录（仅 --quant int8 时需要）')
    p.add_argument('--no-verify', action='store_true',
                   help='跳过 TFLite 验证')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    num_classes, class_names = load_data_yaml(args.data)
    print(f"[export_tflite] data   : {args.data}")
    print(f"[export_tflite] classes: {num_classes}  {class_names}")
    print(f"[export_tflite] quant  : {args.quant}")

    # Step 1: ONNX → SavedModel
    onnx_to_saved_model(args.onnx, args.saved_model_dir)

    # Step 2: SavedModel → TFLite
    # fp16/int8 输出文件名自动加后缀
    output = args.output
    if args.quant != 'none':
        stem, ext = os.path.splitext(output)
        output = f"{stem}_{args.quant}{ext}"

    saved_model_to_tflite(
        args.saved_model_dir, output,
        quant=args.quant,
        calib_dir=args.calib_dir,
        img_size=args.img_size,
    )

    # Step 3: 验证
    if not args.no_verify:
        verify_tflite(output, args.img_size)
