import numpy as np
import tensorflow as tf
from PIL import Image
import os


def load_labels(path):
    """加载 ImageNet 标签文件（如果有的话）"""
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def run_inference(tflite_path, image_path, label_path=None):
    # 1. 加载模型
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 获取模型需要的输入尺寸 (160, 160)
    input_shape = input_details['shape']
    height = input_shape[1]
    width = input_shape[2]

    # 获取量化参数 (Scale 和 Zero Point)
    # 这是一个 Int8 模型，必须要有这两个参数才能正确转换图片
    input_scale, input_zero_point = input_details['quantization']
    print(f"=== 模型参数 ===")
    print(f"Input Shape: {input_shape}")
    print(f"Quantization - Scale: {input_scale}, Zero Point: {input_zero_point}")

    # 2. 图片预处理
    print(f"\n>> 正在读取图片: {image_path}")
    img = Image.open(image_path).convert('RGB')

    # 缩放到模型需要的大小 (160x160)
    img = img.resize((width, height))

    # 转换为 numpy 数组
    input_data = np.array(img, dtype=np.float32)

    # 3. 关键步骤：归一化与量化
    # MCUNet/MobileNet 等通常预处理是 (像素 - 128) / 128，即归一化到 -1 ~ 1 之间
    # 或者是 (像素 / 127.5) - 1.0
    normalized_input = (input_data / 127.5) - 1.0

    # 将归一化后的浮点数转换为模型需要的 int8
    # 公式: q = x / scale + zero_point
    if input_scale > 0:
        q_input = (normalized_input / input_scale) + input_zero_point
        q_input = np.clip(q_input, -128, 127).astype(np.int8)
    else:
        # 如果模型本身不需要量化参数（很少见），直接转
        q_input = normalized_input.astype(np.int8)

    # 增加 Batch 维度: [160, 160, 3] -> [1, 160, 160, 3]
    q_input = np.expand_dims(q_input, axis=0)

    # 4. 运行推理
    interpreter.set_tensor(input_details['index'], q_input)
    interpreter.invoke()

    # 5. 获取结果
    output_data = interpreter.get_tensor(output_details['index'])[0]

    # 如果输出也是 int8，我们需要反量化回浮点数概率（可选，方便观察）
    out_scale, out_zero_point = output_details['quantization']
    if out_scale > 0:
        output_data = (output_data.astype(np.float32) - out_zero_point) * out_scale

    # 6. 打印 Top-5 结果
    top_k = 5
    # argsort 返回的是索引，从大到小排序
    sorted_indices = np.argsort(output_data)[::-1][:top_k]

    labels = load_labels(label_path)

    print(f"\n=== 推理结果 (Top {top_k}) ===")
    for i, idx in enumerate(sorted_indices):
        score = output_data[idx]
        label_name = labels[idx] if labels else f"Class ID {idx}"
        print(f"{i + 1}. [{label_name}] \t(Score: {score:.4f}, ID: {idx})")


if __name__ == "__main__":
    # --- 配置区域 ---
    MODEL_FILE = "/home/verser/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.tflite"  # 替换为你的 tflite 路径
    IMAGE_FILE = "/home/verser/Python/GD_Net/assets/ILSVRC2017_test_00000001.JPEG"  # 替换为你的一张测试图片路径
    LABEL_FILE = "imagenet_labels.txt"  # 可选：ImageNet 标签文件路径

    if not os.path.exists(IMAGE_FILE):
        print(f"❌ 请准备一张测试图片并命名为 {IMAGE_FILE}")
    else:
        run_inference(MODEL_FILE, IMAGE_FILE, LABEL_FILE)