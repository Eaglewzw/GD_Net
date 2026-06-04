import argparse
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from tqdm import tqdm

from gd_net.model import YOLOv3_McuNet
from utils.augmentations import letterbox
from cfg import cfg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ==================== 工具函数：IoU 矩阵计算 (Numpy版) ====================
def box_iou(box1, box2):
    """
    计算两个框列表之间的 IoU
    box1: [N, 4] (x1, y1, x2, y2)
    box2: [M, 4] (x1, y1, x2, y2)
    返回: [N, M]
    """

    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    return inter / union


# ==================== 核心评估逻辑 ====================
def compute_ap(recall, precision):
    """ 计算 AP (使用 All-Points 插值，COCO标准) """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    x = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[x + 1] - mrec[x]) * mpre[x + 1])

    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    计算单类别的 P, R, AP, F1
    tp: [N, 10] bool矩阵
    """
    # 按照置信度降序排列
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)

    ap, p, r = [], [], []

    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = (tp[i]).cumsum(0)

        recall = tpc / (n_gt + 1e-16)
        r.append(recall[:, 0])  # 取 IoU=0.5 的 recall

        precision = tpc / (tpc + fpc)
        p.append(precision[:, 0])  # 取 IoU=0.5 的 precision

        ap.append([compute_ap(recall[:, j], precision[:, j]) for j in range(tp.shape[1])])

    return p, r, ap, unique_classes


# ==================== 数据解析 ====================
def parse_voc_xml(xml_path, class_names=["drone"]):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name not in class_names:
            continue
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax, 0])  # class_id 0 for drone
    return np.array(boxes) if len(boxes) > 0 else np.zeros((0, 5))


# ==================== 主评估函数 ====================
@torch.no_grad()
def evaluate_dataset(args):
    img_dir = os.path.join(args.data_root, 'JPEGImages')
    ann_dir = os.path.join(args.data_root, 'Annotations')
    val_list_path = os.path.join(args.data_root, 'ImageSets/Main/val.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(val_list_path):
        logger.error(f"验证列表不存在: {val_list_path}")
        return

    with open(val_list_path, 'r') as f:
        img_ids = [line.strip() for line in f.readlines()]

    logger.info("正在加载模型权重...")
    if not os.path.exists(args.weights):
        logger.error(f"Checkpoint 不存在: {args.weights}")
        return

    ckpt = torch.load(args.weights, map_location='cpu')
    model = YOLOv3_McuNet(cfg, device=device, num_classes=1, trainable=False, deploy=False)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    logger.info(f"模型加载完成，运行设备: {device}")

    stats = []
    iouv = np.linspace(0.5, 0.95, 10)
    niou = iouv.size

    for img_id in tqdm(img_ids, desc="Evaluating"):
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        ann_path = os.path.join(ann_dir, f"{img_id}.xml")

        import cv2
        if not os.path.exists(img_path):
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        labels = np.zeros((0, 5))
        if os.path.exists(ann_path):
            labels = parse_voc_xml(ann_path)

        x, ratio, (dw, dh) = letterbox(frame, new_shape=args.img_size)
        x = x.to(device)
        outputs = model(x)

        bboxes = outputs['bboxes']
        scores = outputs['scores']
        pred_labels = outputs['labels']

        keep = scores > args.conf_thres
        bboxes = bboxes[keep]
        scores = scores[keep]
        pred_labels = pred_labels[keep]

        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / ratio
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / ratio

        nl = len(labels)
        npr = len(bboxes)
        correct = np.zeros((npr, niou), dtype=bool)

        if npr == 0:
            if nl:
                stats.append((correct, *np.zeros((2, 0)), labels[:, 4]))
            continue

        if nl:
            tbox = labels[:, :4]
            iou = box_iou(bboxes, tbox)

            for i in range(niou):
                matches = np.nonzero(iou >= iouv[i])
                matches = np.array(matches).T

                if matches.shape[0]:
                    matches_iou = iou[matches[:, 0], matches[:, 1]]
                    matches = matches[np.argsort(matches_iou)[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    correct[matches[:, 0].astype(int), i] = True

        stats.append((correct, scores, pred_labels, labels[:, 4]))

    # ---------------- 计算指标 ----------------
    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    if len(stats) and stats[0].any():
        p, r, ap, ap_class = ap_per_class(*stats)

        ap = np.array(ap)[0]  # [10]
        p_curve = np.array(p)[0]
        r_curve = np.array(r)[0]

        map_05 = ap[0]
        map_05_95 = ap.mean()

        f1 = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
        i = f1.argmax()
        best_p = p_curve[i]
        best_r = r_curve[i]
        best_f1 = f1[i]

        logger.info("\n" + "=" * 80)
        logger.info(
            f"{'Class':<10} {'Images':<10} {'Targets':<10} {'P (Best F1)':<15} "
            f"{'R (Best F1)':<15} {'mAP@.5':<10} {'mAP@.5:.95':<10}")
        logger.info("-" * 80)
        logger.info(
            f"{'Drone':<10} {len(img_ids):<10} {len(stats[3]):<10} "
            f"{best_p:.4f}          {best_r:.4f}          {map_05:.4f}     {map_05_95:.4f}")
        logger.info("=" * 80)
        logger.info(f"Best F1 Score: {best_f1:.4f}")

    else:
        logger.warning("No detections or ground truths found!")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv3-McuNet VOC mAP Evaluation")
    parser.add_argument('--data-root', type=str,
                        default='/media/verser/robot/Dataset/DT_Drone',
                        help='数据集根目录')
    parser.add_argument('--weights', type=str,
                        default='./checkpoints/gd_net_l.pth',
                        help='模型权重路径')
    parser.add_argument('--img-size', type=int, default=320,
                        help='推理图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.1,
                        help='置信度阈值（评估时建议设低以保证 Recall 准确）')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate_dataset(parse_args())
