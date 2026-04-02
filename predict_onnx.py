# detect_video_onnx_fps_with_nms.py
# 完整 ONNX 推理 + 实时 FPS + 纯 numpy 版 NMS（无需 torchvision）

import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# ======================== 可视化 & rescale ========================
try:
    from utils.ops import rescale_bboxes
    from utils.plotting import visualize
except:
    def visualize(image, bboxes, scores, labels, class_colors, class_names, class_indexs):
        img = image.copy()
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = map(int, bbox)
            color = class_colors[int(label)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            txt = f'{class_names[int(label)]} {score:.2f}'
            cv2.putText(img, txt, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return img

    def rescale_bboxes(bboxes, orig_size, ratio, pad=None):
        orig_w, orig_h = orig_size
        if pad is not None:
            dw, dh = pad
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - dh
        bboxes /= ratio
        return bboxes

# ======================== 纯 numpy 版 NMS（关键！）====================
def nms(boxes, scores, iou_threshold=0.45):
    """
    纯 numpy 实现 NMS
    boxes: [N, 4] (x1, y1, x2, y2)
    scores: [N]
    返回: keep 索引列表
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 从高到低排序
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# ======================== 手动 letterbox ========================
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw - 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, r, (dw, dh)

# ======================== 主推理函数（已加 NMS）====================
def detect_video_onnx(video_path, save_path, onnx_path, conf_thresh=0.3, nms_thresh=0.45):
    os.makedirs(save_path, exist_ok=True)

    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX 模型加载成功: {onnx_path}")
    print(f"   输入名: {input_name} | 输出名: {output_name}")
    print(f"   推理引擎: {sess.get_providers()}")

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"原视频分辨率: {width}×{height}  FPS: {fps_video:.1f}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_name = os.path.join(save_path, os.path.basename(video_path).split('.')[0] + '_det_onnx_nms.avi')
    out = cv2.VideoWriter(save_name, fourcc, 30.0, (1920, 1080))

    np.random.seed(0)
    class_colors = [(0, 255, 255)]
    class_names = ["drone"]

    frame_id = 0
    fps_list = []
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        orig_h, orig_w = frame.shape[:2]

        x, ratio, (dw, dh) = letterbox(frame, new_shape=640)

        t0 = time.time()
        outputs = sess.run([output_name], {input_name: x})[0]
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000

        # FPS 计算
        curr_time = time.time()
        fps_raw = 1.0 / (curr_time - prev_time) if frame_id > 1 else 0
        fps_list.append(fps_raw)
        if len(fps_list) > 30:
            fps_list.pop(0)
        fps_smooth = sum(fps_list) / len(fps_list) if fps_list else 0
        prev_time = curr_time

        # 关键：置信度过滤 + NMS
        outputs = outputs[outputs[:, 4] > conf_thresh]
        if len(outputs) == 0:
            det_text = "No drone"
            bboxes = scores = labels = np.array([])
        else:
            boxes = outputs[:, :4]
            scores = outputs[:, 4]
            labels = outputs[:, 5].astype(int) if outputs.shape[1] > 5 else np.zeros(len(outputs), dtype=int)

            # 手动 NMS（解决重叠框！）
            keep = nms(boxes, scores, iou_threshold=nms_thresh)
            outputs = outputs[keep]

            bboxes = outputs[:, :4]
            scores = outputs[:, 4]
            labels = outputs[:, 5].astype(int) if outputs.shape[1] > 5 else np.zeros(len(outputs), dtype=int)
            det_text = f"{len(bboxes)} drone{'s' if len(bboxes)>1 else ''}"

            # 坐标还原
            bboxes[:, [0, 2]] -= dw
            bboxes[:, [1, 3]] -= dh
            bboxes /= ratio

        # 可视化
        frame_vis = visualize(frame, bboxes, scores, labels, class_colors, class_names, [0])
        frame_show = cv2.resize(frame_vis, (1920, 1080))

        # 叠加信息
        cv2.putText(frame_show, f"FPS: {fps_smooth:.1f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4)
        cv2.putText(frame_show, f"Infer: {infer_ms:.1f}ms", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
        cv2.putText(frame_show, det_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

        out.write(frame_show)
        cv2.imshow('ONNX Drone Detection + NMS - Press Q', frame_show)

        print(f"Frame {frame_id:4d} | FPS: {fps_smooth:5.1f} | Infer: {infer_ms:5.1f}ms | {det_text}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n检测完成！视频已保存: {save_name}")
    print(f"平均 FPS: {sum(fps_list)/len(fps_list):.1f}" if fps_list else "")

# ============================== 运行 ==============================
if __name__ == '__main__':
    video_path = '/media/verser/robot/Dataset/ARD-MAV/videos/phantom16.mp4'
    save_path  = './det_results_onnx/'
    onnx_path  = './checkpoints/yolov3_mcu.onnx'

    detect_video_onnx(
        video_path=video_path,
        save_path=save_path,
        onnx_path=onnx_path,
        conf_thresh=0.3,     # 置信度阈值（建议 0.25~0.4）
        nms_thresh=0.45      # NMS IoU 阈值（0.4~0.5 最佳）
    )