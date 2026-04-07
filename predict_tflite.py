"""
predict_tflite.py  ── 使用 TFLite 模型对图片或视频做推理

TFLite 模型输出格式（与 ONNX deploy 模式一致）：[N_anchors, 4+C]
    前 4 列: x1 y1 x2 y2（letterbox 坐标系）
    后 C 列: 每类联合置信度 sqrt(obj * cls)

    注意 INT8 量化模型输出为 int8，推理时自动反量化为 float32。

用法（图片）：
    CUDA_VISIBLE_DEVICES="" python predict_tflite.py \
        --source path/to/img.jpg \
        --tflite ./checkpoints/yolov3_mcu.tflite \
        --data data/standford.yaml

用法（视频）：
    CUDA_VISIBLE_DEVICES="" python predict_tflite.py \
        --source path/to/video.mp4 \
        --tflite ./checkpoints/yolov3_mcu.tflite \
        --data data/standford.yaml
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = ''   # TFLite 推理只用 CPU，避免 CUDA 冲突


# ──────────────────────────────────────────────────────────────
#  图像预处理（与 predict_onnx.py 完全一致）
# ──────────────────────────────────────────────────────────────
def letterbox(img_bgr, new_shape=320, color=(114, 114, 114)):
    h, w = img_bgr.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = (new_shape - new_w) // 2
    dh = (new_shape - new_h) // 2

    img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis], r, (dw, dh)   # [1, H, W, 3] NHWC


# ──────────────────────────────────────────────────────────────
#  后处理（与 predict_onnx.py 完全一致）
# ──────────────────────────────────────────────────────────────
def nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def postprocess(raw, ratio, dw, dh, conf_thresh, nms_thresh, num_classes):
    """raw: [N_anchors, 4+C]，letterbox 坐标系"""
    boxes_lb  = raw[:, :4]
    cls_scores = raw[:, 4:]

    labels = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(labels)), labels]

    mask = scores > conf_thresh
    if not mask.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    boxes_lb, scores, labels = boxes_lb[mask], scores[mask], labels[mask]

    keep_all = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep = nms(boxes_lb[idx], scores[idx], nms_thresh)
        keep_all.extend(idx[keep].tolist())

    boxes_lb = boxes_lb[keep_all]
    scores   = scores[keep_all]
    labels   = labels[keep_all]

    boxes = boxes_lb.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

    return boxes, scores, labels.astype(int)


# ──────────────────────────────────────────────────────────────
#  可视化
# ──────────────────────────────────────────────────────────────
_PALETTE = [
    (0, 255, 255), (0, 128, 255), (0, 255, 128),
    (255, 128, 0), (255, 0, 128), (128, 0, 255),
]


def draw_boxes(img, boxes, scores, labels, class_names):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = _PALETTE[int(label) % len(_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        cv2.putText(img, f'{name} {score:.2f}', (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


# ──────────────────────────────────────────────────────────────
#  TFLite 推理器
# ──────────────────────────────────────────────────────────────
class TFLiteDetector:
    def __init__(self, tflite_path, img_size=320, num_classes=1,
                 conf_thresh=0.25, nms_thresh=0.45, class_names=None):
        import tensorflow as tf

        self.img_size    = img_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]

        self.interp = tf.lite.Interpreter(model_path=tflite_path)
        self.interp.allocate_tensors()

        self.inp_detail = self.interp.get_input_details()[0]
        self.out_detail = self.interp.get_output_details()[0]
        self.is_int8    = self.inp_detail['dtype'] == np.int8

        print(f"[TFLiteDetector] Loaded: {tflite_path}")
        print(f"  input : {self.inp_detail['name']}  "
              f"shape={self.inp_detail['shape']}  dtype={self.inp_detail['dtype']}")
        print(f"  output: {self.out_detail['name']}  "
              f"shape={self.out_detail['shape']}  dtype={self.out_detail['dtype']}")
        print(f"  INT8 mode: {self.is_int8}")

        # warmup
        dummy = np.zeros(self.inp_detail['shape'],
                         dtype=np.int8 if self.is_int8 else np.float32)
        self.interp.set_tensor(self.inp_detail['index'], dummy)
        self.interp.invoke()
        print("  warmup done ✓")

    def _quantize_input(self, x_fp32):
        """FP32 → INT8（仅 INT8 模型使用）"""
        scale, zp = self.inp_detail['quantization']
        return (x_fp32 / scale + zp).clip(-128, 127).astype(np.int8)

    def _dequantize_output(self, x_int8):
        """INT8 → FP32"""
        scale, zp = self.out_detail['quantization']
        return (x_int8.astype(np.float32) - zp) * scale

    def infer(self, frame_bgr):
        # 预处理：返回 NHWC float32
        blob, ratio, (dw, dh) = letterbox(frame_bgr, self.img_size)

        # INT8 模型需要量化输入
        inp = self._quantize_input(blob) if self.is_int8 else blob

        t0 = time.perf_counter()
        self.interp.set_tensor(self.inp_detail['index'], inp)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out_detail['index'])
        infer_ms = (time.perf_counter() - t0) * 1000

        # INT8 输出反量化
        if self.is_int8:
            raw = self._dequantize_output(raw)

        # raw: [1, N_anchors, 4+C] 或 [N_anchors, 4+C]，统一降维
        if raw.ndim == 3:
            raw = raw[0]

        boxes, scores, labels = postprocess(
            raw, ratio, dw, dh,
            self.conf_thresh, self.nms_thresh, self.num_classes
        )

        vis = draw_boxes(frame_bgr.copy(), boxes, scores, labels, self.class_names)
        return vis, boxes, scores, labels, infer_ms


# ──────────────────────────────────────────────────────────────
#  图片 / 视频处理
# ──────────────────────────────────────────────────────────────
IMG_EXTS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
VID_EXTS = {'.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv'}


class FPSCounter:
    def __init__(self, window=60):
        self._t = []
        self._w = window

    def tick(self):
        self._t.append(time.perf_counter())
        if len(self._t) > self._w:
            self._t.pop(0)
        return len(self._t) / (self._t[-1] - self._t[0]) if len(self._t) >= 2 else 0.0


def run_image(detector, src, save_dir):
    frame = cv2.imread(str(src))
    if frame is None:
        print(f"[error] Cannot read image: {src}"); return

    vis, boxes, scores, labels, ms = detector.infer(frame)
    cv2.putText(vis, f"Objects:{len(boxes)}  Infer:{ms:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    dst = os.path.join(save_dir, Path(src).name)
    cv2.imwrite(dst, vis)
    print(f"[image] {len(boxes)} detections | {ms:.1f}ms → saved {dst}")


def run_video(detector, src, save_dir):
    cap = cv2.VideoCapture(str(src))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dst = os.path.join(save_dir, Path(src).stem + '_tflite.mp4')
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (W, H))

    fps_ctr = FPSCounter()
    fid = 0

    cv2.namedWindow('TFLite Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('TFLite Detection', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1

        vis, boxes, scores, labels, ms = detector.infer(frame)
        fps_real = fps_ctr.tick()

        cv2.putText(vis, f"FPS:{fps_real:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(vis, f"Infer:{ms:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, f"Det:{len(boxes)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        writer.write(vis)
        cv2.imshow('TFLite Detection', vis)
        print(f"\rFrame {fid}/{total} | FPS {fps_real:.1f} | Det {len(boxes)} | {ms:.1f}ms",
              end='', flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print()
            break

    print(f"\n[video] saved → {dst}")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='TFLite Inference')
    p.add_argument('--source', type=str,
                   default='/home/verser/Pictures/00019.jpg')
    p.add_argument('--tflite', type=str,
                   default='./checkpoints/yolov3_mcu.tflite')
    p.add_argument('--data', type=str,
                   default='./data/standford.yaml',
                   help='数据集 yaml，自动读取类别数和类别名')
    p.add_argument('--output', type=str, default='./det_results_tflite/')
    p.add_argument('--img-size', type=int, default=320)
    p.add_argument('--conf-thresh', type=float, default=0.25)
    p.add_argument('--nms-thresh', type=float, default=0.45)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.data, 'r') as f:
        _d = yaml.safe_load(f)
    names_dict  = _d.get('names', {0: 'object'})
    class_names = [names_dict[i] for i in sorted(names_dict)]
    num_classes = len(class_names)

    detector = TFLiteDetector(
        tflite_path=args.tflite,
        img_size=args.img_size,
        num_classes=num_classes,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        class_names=class_names,
    )

    src = Path(args.source)
    if src.suffix.lower() in IMG_EXTS:
        run_image(detector, src, args.output)
    elif src.suffix.lower() in VID_EXTS:
        run_video(detector, src, args.output)
    else:
        print(f"[error] Unsupported file type: {src.suffix}")
