import torch
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from gd_net.yolov3_mcu_net import YOLOv3_McuNet
from yolov3_mcu_config import cfg
from argparse import Namespace


# ==================== 正确的 letterbox（兼容 PyTorch 2.4+） ====================
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw // 2, dh // 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # BGR → RGB, HWC → CHW, /255, add batch
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor, r, (dw, dh)


# ==================== 计算 IoU ====================
def compute_iou(box1, box2):
    """box: [xmin, ymin, xmax, ymax]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union if union > 0 else 0


# ==================== 计算 AP (VOC 2007 风格: 11-point interpolation) ====================
def compute_ap(recalls, precisions):
    """recalls and precisions are sorted lists"""
    ap = 0.0
    for i in range(11):
        r = i / 10.0
        p = max([precisions[j] for j in range(len(recalls)) if recalls[j] >= r], default=0)
        ap += p / 11.0
    return ap


# ==================== 计算 mAP (单类，所以 mAP = AP) ====================
def evaluate_map(preds, gts, iou_thres=0.5, class_id=0):
    """
    preds: dict {img_id: list[[xmin, ymin, xmax, ymax, score], ...]}
    gts: dict {img_id: list[[xmin, ymin, xmax, ymax, difficult], ...]}
    """
    all_preds = []
    total_gts = 0
    for img_id in gts:
        gt_boxes = gts[img_id]
        total_gts += len([gt for gt in gt_boxes if gt[4] == 0])  # 非difficult

        if img_id in preds:
            for pred in preds[img_id]:
                all_preds.append((pred[4], img_id, pred[:4]))  # (score, img_id, box)

    # 按 score 降序排序
    all_preds.sort(key=lambda x: x[0], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    detected = {img_id: set() for img_id in gts}

    for i, (score, img_id, pred_box) in enumerate(all_preds):
        if img_id not in gts:
            fp[i] = 1
            continue

        max_iou = 0
        max_idx = -1
        for j, gt_box in enumerate(gts[img_id]):
            if j in detected[img_id] or gt_box[4] == 1:  # 已检测 or difficult
                continue
            iou = compute_iou(pred_box, gt_box[:4])
            if iou > max_iou:
                max_iou = iou
                max_idx = j

        if max_iou >= iou_thres:
            tp[i] = 1
            detected[img_id].add(max_idx)
        else:
            fp[i] = 1

    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    recalls = tp_cum / total_gts if total_gts > 0 else np.zeros(len(tp_cum))
    precisions = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum).any() else np.zeros(len(tp_cum))

    return compute_ap(recalls.tolist(), precisions.tolist())


# ==================== 解析 VOC XML ====================
def parse_voc_xml(xml_path, class_names=["drone"]):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name not in class_names:
            continue
        difficult = int(obj.find('difficult').text) if obj.find('difficult') else 0
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax, difficult])
    return boxes


# ==================== 主评估函数 ====================
def evaluate_dataset():
    # 数据集路径（VOC格式）
    dataset_root = '/media/verser/robot/Dataset/DT_Drone'  # 替换成你的路径
    img_dir = os.path.join(dataset_root, 'JPEGImages')
    ann_dir = os.path.join(dataset_root, 'Annotations')
    val_list_path = os.path.join(dataset_root, 'imageSets/Main/val.txt')  # val.txt 路径

    # 读取 val 图像 ID 列表
    with open(val_list_path, 'r') as f:
        img_ids = [line.strip() for line in f.readlines()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 640

    # ==================== 模型加载 ====================
    print("正在加载模型权重...")
    ckpt = torch.load('/home/verser/Python/GD_Net/checkpoints/best_yolov3_mcu.pth', map_location='cpu')

    model = YOLOv3_McuNet(
        cfg,
        device=device,
        num_classes=1,
        trainable=False,
        deploy=False,
    )

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    print(f"模型加载完成！运行设备: {device}")

    # 收集 preds 和 gts
    preds = {}  # {img_id: [[xmin, ymin, xmax, ymax, score], ...]}
    gts = {}  # {img_id: [[xmin, ymin, xmax, ymax, difficult], ...]}

    for img_id in tqdm(img_ids, desc="Evaluating"):
        img_path = os.path.join(img_dir, f"{img_id}.jpg")  # 假设 jpg
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # 预处理
        x, ratio, (dw, dh) = letterbox(frame, new_shape=img_size)
        x = x.to(device)

        # 推理
        with torch.no_grad():
            outputs = model(x)

        bboxes = outputs['bboxes']
        scores = outputs['scores']
        labels = outputs['labels']

        if len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / ratio
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / ratio

            # 过滤 labels (只取 class 0)
            mask = labels == 0
            bboxes = bboxes[mask]
            scores = scores[mask]

            # 组合成 [xmin, ymin, xmax, ymax, score]
            dets = np.hstack((bboxes, scores.reshape(-1, 1)))
            preds[img_id] = dets.tolist()

        # 加载 ground truth
        ann_path = os.path.join(ann_dir, f"{img_id}.xml")
        if os.path.exists(ann_path):
            gts[img_id] = parse_voc_xml(ann_path)

    # 计算 mAP
    ap = evaluate_map(preds, gts, iou_thres=0.5)
    print(f"mAP@0.5: {ap:.4f}")


if __name__ == '__main__':
    evaluate_dataset()