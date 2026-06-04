import os
import csv
import math
import random
import argparse
import logging

import yaml
import numpy as np
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

from gd_net.model import YOLOv3_McuNet
from gd_net.dataset import YoloDataset, DataLoader
from gd_net.loss import build_criterion
from cfg import cfg

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """固定所有随机种子，保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets


# ==================== 评估工具函数 ====================

def _box_iou_np(box1, box2):
    """计算两组框的 IoU 矩阵，box1: [N,4], box2: [M,4], 返回 [N,M]"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    inter = (rb - lt).clip(min=0).prod(axis=2)
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def _compute_ap(recall, precision):
    """All-points 插值计算 AP"""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate_and_plot(model, dataloader, device, num_classes, save_dir, class_names=None):
    """
    在训练集上跑一遍推理，计算每类的 Precision / Recall / F1 / AP@0.5，
    并保存 PR 曲线 + F1 曲线 + 指标汇总表图像。
    """
    model.eval()
    model.trainable = False   # 切换到推理分支，返回 {bboxes, scores, labels}
    iou_thres = 0.5
    stats = []   # list of (correct[bool], conf, pred_cls, gt_cls)

    for imgs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        imgs = imgs.to(device)
        for b_idx in range(imgs.shape[0]):
            x = imgs[b_idx:b_idx + 1]
            outputs = model(x)

            bboxes     = outputs['bboxes']      # np [N,4]
            scores     = outputs['scores']      # np [N]
            pred_lbls  = outputs['labels']      # np [N]

            gt_boxes  = targets[b_idx]['boxes'].numpy()   # [M,4]
            gt_labels = targets[b_idx]['labels'].numpy()  # [M]

            npr = len(bboxes)
            nl  = len(gt_boxes)
            correct = np.zeros((npr,), dtype=bool)

            if npr == 0:
                if nl:
                    stats.append((correct, np.array([]), np.array([]), gt_labels))
                continue

            if nl:
                iou = _box_iou_np(bboxes, gt_boxes)          # [N, M]
                matched_gt = set()
                for pi in range(npr):
                    best_iou, best_gi = -1, -1
                    for gi in range(nl):
                        if gi in matched_gt:
                            continue
                        if iou[pi, gi] > best_iou:
                            best_iou, best_gi = iou[pi, gi], gi
                    if best_iou >= iou_thres and pred_lbls[pi] == gt_labels[best_gi]:
                        correct[pi] = True
                        matched_gt.add(best_gi)

            stats.append((correct, scores, pred_lbls, gt_labels))

    if not stats:
        logger.warning("评估：没有收集到任何统计数据，跳过绘图。")
        model.train()
        return

    all_correct  = np.concatenate([s[0] for s in stats])
    all_conf     = np.concatenate([s[1] for s in stats])
    all_pred_cls = np.concatenate([s[2] for s in stats])
    all_gt_cls   = np.concatenate([s[3] for s in stats])

    # 按置信度降序排列
    sort_idx    = np.argsort(-all_conf)
    all_correct  = all_correct[sort_idx]
    all_conf     = all_conf[sort_idx]
    all_pred_cls = all_pred_cls[sort_idx]

    unique_classes = np.unique(all_gt_cls).astype(int)
    if class_names is None:
        class_names = {c: f"class_{c}" for c in unique_classes}

    results = {}   # cls_id -> {p_curve, r_curve, f1_curve, ap, conf_curve}
    for c in unique_classes:
        mask  = all_pred_cls == c
        n_gt  = int((all_gt_cls == c).sum())
        n_p   = int(mask.sum())

        if n_p == 0 or n_gt == 0:
            results[c] = dict(p=0., r=0., f1=0., ap=0.,
                              p_curve=np.array([0., 1.]),
                              r_curve=np.array([0., 0.]),
                              f1_curve=np.array([0., 0.]),
                              conf_curve=np.array([1., 0.]))
            continue

        tp = all_correct[mask].astype(float)
        fp = 1.0 - tp
        conf_c = all_conf[mask]

        tpc = tp.cumsum()
        fpc = fp.cumsum()
        recall_c    = tpc / (n_gt + 1e-16)
        precision_c = tpc / (tpc + fpc + 1e-16)
        f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-16)

        ap = _compute_ap(recall_c, precision_c)
        best_i = int(f1_c.argmax())

        results[c] = dict(
            p=float(precision_c[best_i]),
            r=float(recall_c[best_i]),
            f1=float(f1_c[best_i]),
            ap=ap,
            p_curve=precision_c,
            r_curve=recall_c,
            f1_curve=f1_c,
            conf_curve=conf_c,
        )

    # ---- 绘图 ----
    n_cls = len(unique_classes)
    big = n_cls > 20  # 大类阈值：超过则启用自适应布局

    # 颜色：tab10 → tab20 → hsv，避免大类时撞色
    if n_cls <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_cls, 1)))
    elif n_cls <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_cls))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, n_cls, endpoint=False))

    # 布局：≤20 类沿用原 1×3；>20 类改为上下两行，下排表格分多列
    if big:
        rows_per_col = 20
        n_table_cols = math.ceil((n_cls + 1) / rows_per_col)  # +1 = mean 行
        per_col      = math.ceil((n_cls + 1) / n_table_cols)
        fig_w  = max(14.0, n_table_cols * 4.0)
        tbl_h  = per_col * 0.28 + 1.2
        fig    = plt.figure(figsize=(fig_w, 5 + tbl_h))
        gs     = fig.add_gridspec(2, 2, height_ratios=[5, tbl_h], hspace=0.3)
        ax_pr  = fig.add_subplot(gs[0, 0])
        ax_f1  = fig.add_subplot(gs[0, 1])
        ax_tbl = fig.add_subplot(gs[1, :])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)
        ax_pr, ax_f1, ax_tbl = axes

    # -- PR 曲线 --
    for i, c in enumerate(unique_classes):
        r = results[c]
        name = class_names.get(int(c), f"cls_{c}")
        ax_pr.plot(r['r_curve'], r['p_curve'],
                   color=colors[i], linewidth=1.5,
                   label=f"{name} AP={r['ap']:.3f}")
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.05)
    ax_pr.grid(True, linestyle='--', alpha=0.5)
    if not big:
        ax_pr.legend(loc='lower left', fontsize=8)

    # -- F1 曲线 --
    for i, c in enumerate(unique_classes):
        r = results[c]
        name = class_names.get(int(c), f"cls_{c}")
        ax_f1.plot(r['conf_curve'], r['f1_curve'],
                   color=colors[i], linewidth=1.5,
                   label=f"{name} F1={r['f1']:.3f}")
    ax_f1.set_xlabel('Confidence Threshold')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.set_title('F1-Confidence Curve')
    ax_f1.set_xlim(0, 1); ax_f1.set_ylim(0, 1.05)
    ax_f1.grid(True, linestyle='--', alpha=0.5)
    if not big:
        ax_f1.legend(loc='upper right', fontsize=8)

    # -- 指标汇总表 --
    ax_tbl.axis('off')
    col_labels = ['Class', 'Precision', 'Recall', 'F1', 'AP@0.5']
    rows = []
    for c in unique_classes:
        r = results[c]
        name = class_names.get(int(c), f"cls_{c}")
        rows.append([name, f"{r['p']:.4f}", f"{r['r']:.4f}",
                     f"{r['f1']:.4f}", f"{r['ap']:.4f}"])
    mean_p  = np.mean([results[c]['p']  for c in unique_classes])
    mean_r  = np.mean([results[c]['r']  for c in unique_classes])
    mean_f1 = np.mean([results[c]['f1'] for c in unique_classes])
    mean_ap = np.mean([results[c]['ap'] for c in unique_classes])
    rows.append(['mean', f"{mean_p:.4f}", f"{mean_r:.4f}",
                 f"{mean_f1:.4f}", f"{mean_ap:.4f}"])

    if big:
        # 将 rows 横向拼成 n_table_cols 个子表，每个子表 per_col 行
        merged_labels = col_labels * n_table_cols
        merged_rows = []
        for r_i in range(per_col):
            row = []
            for col_i in range(n_table_cols):
                idx = col_i * per_col + r_i
                if idx < len(rows):
                    row.extend(rows[idx])
                else:
                    row.extend([''] * len(col_labels))
            merged_rows.append(row)
        tbl = ax_tbl.table(cellText=merged_rows, colLabels=merged_labels,
                           loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.2)
        # 高亮均值行（位于最后一行 → 定位它在哪个子列）
        mean_idx       = len(rows) - 1
        mean_sub_col   = mean_idx // per_col
        mean_table_row = mean_idx %  per_col + 1  # +1 跳过表头
        for k in range(len(col_labels)):
            tbl[mean_table_row,
                mean_sub_col * len(col_labels) + k].set_facecolor('#d0e8ff')
    else:
        tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                           loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        for col in range(len(col_labels)):
            tbl[len(rows), col].set_facecolor('#d0e8ff')
    ax_tbl.set_title('Metrics Summary', fontsize=12, pad=12)

    save_path = os.path.join(save_dir, "eval_metrics.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 全量指标同步写 CSV，PNG 永远只做快览
    csv_path = os.path.join(save_dir, "eval_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(col_labels)
        for row in rows:
            writer.writerow(row)

    logger.info(f"评估图表已保存至 {save_path}")
    logger.info(f"评估指标 CSV 已保存至 {csv_path}")
    logger.info(f"  mAP@0.5={mean_ap:.4f}  P={mean_p:.4f}  R={mean_r:.4f}  F1={mean_f1:.4f}")

    model.trainable = True
    model.train()


def load_data_cfg(yaml_path):
    """从 yaml 文件读取数据集配置，返回 (img_dir, label_dir, label_format,
       num_classes, class_mapping, class_names)"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data_root    = data['path']
    label_format = data.get('label_format', 'voc')
    class_names  = data.get('names', {})

    if label_format == 'yolo':
        num_classes = data.get('num_classes', len(class_names))
        class_mapping = data.get('class_mapping', {i: i for i in range(num_classes)})
    else:
        class_mapping = data['class_mapping']
        num_classes   = len(set(class_mapping.values()))

    # 路径：支持绝对路径和相对路径（相对 path）
    img_dir   = data.get('img_dir', 'JPEGImages')
    label_dir = data.get('label_dir', 'Annotations')
    if not os.path.isabs(img_dir):
        img_dir = os.path.join(data_root, img_dir)
    if not os.path.isabs(label_dir):
        label_dir = os.path.join(data_root, label_dir)

    logger.info(f"数据集配置: {yaml_path}")
    logger.info(f"  图片目录: {img_dir}")
    logger.info(f"  标签目录: {label_dir}")
    logger.info(f"  标签格式: {label_format}")
    logger.info(f"  类别数: {num_classes}  {list(class_names.values())[:10]}{'...' if len(class_names) > 10 else ''}")
    return img_dir, label_dir, label_format, num_classes, class_mapping, class_names


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv3-McuNet Training")
    # 数据集配置 yaml（优先级高于 --data-root / --num-classes）
    parser.add_argument('--data', type=str, default='/home/verser/Python/GD_Net/data/coco.yaml',
                        help='数据集配置文件，如 data/udacity.yaml')
    # 兼容旧用法（当未指定 --data 时生效）
    parser.add_argument('--data-root', type=str,
                        default='/media/verser/robot/Dataset/UdacitySelfDrivingCarDataset/Driving_Car',
                        help='数据集根目录（--data 未指定时使用）')
    # 检查点保存目录
    parser.add_argument('--save-dir', type=str,
                        default='./checkpoints',
                        help='模型保存目录')
    # 预训练权重路径（可选）
    parser.add_argument('--pretrained', type=str,
                        default='/home/verser/Python/GD_Net/checkpoints/best_yolov3_mcu.pth',
                        help='预训练权重路径，留空则从头训练')
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    # 必须是32的倍数, cfg.py中max_stride: 32，特征图需要整除
    parser.add_argument('--img-size', type=int, default=320)  #必须是32的倍数
    parser.add_argument('--num-classes', type=int, default=5,
                        help='类别数（--data 未指定时使用）')
    parser.add_argument('--lr', type=float, default=0.012)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=int, default=0, help='CUDA 设备编号')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ------------------- 设备 -------------------
    torch.cuda.set_device(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ------------------- 数据集配置 -------------------
    if args.data:
        img_dir, label_dir, label_format, num_classes, class_mapping, class_names_map = \
            load_data_cfg(args.data)
    else:
        img_dir       = os.path.join(args.data_root, 'JPEGImages')
        label_dir     = os.path.join(args.data_root, 'Annotations')
        label_format  = 'voc'
        num_classes   = args.num_classes
        class_mapping = None
        class_names_map = None

    dataset = YoloDataset(img_dir=img_dir, label_dir=label_dir,
                          img_size=args.img_size,
                          transform=torchvision.transforms.ToTensor(),
                          class_mapping=class_mapping,
                          label_format=label_format)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=collate_fn,
                            persistent_workers=True, pin_memory=True)

    # ------------------- 模型 -------------------
    model = YOLOv3_McuNet(cfg, device, num_classes=num_classes, trainable=True).to(device)

    pretrained_path = args.pretrained or os.path.join(args.save_dir, 'best_yolov3_mcu.pth')
    if os.path.exists(pretrained_path):
        logger.info(f"加载预训练模型: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
            logger.info("预训练权重加载成功")
        except Exception as e:
            logger.warning(f"加载预训练权重失败: {e}，将从头开始训练")
    else:
        logger.info("未找到预训练权重，将从头开始训练")

    # ------------------- 损失 / 优化器 / 调度器 -------------------
    criterion = build_criterion(cfg, device, num_classes=num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    total_steps = len(dataloader) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                           pct_start=0.1, div_factor=300, final_div_factor=1e5,
                           anneal_strategy='cos')

    # ------------------- 训练准备 -------------------
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss, best_epoch = float('inf'), 0

    epoch_list, loss_total_list = [], []
    loss_obj_list, loss_cls_list, loss_box_list = [], [], []

    # 初始化 loss CSV，每个 epoch 实时写入
    loss_csv_path = os.path.join(args.save_dir, "loss_history.csv")
    loss_csv = open(loss_csv_path, 'w', newline='')
    loss_writer = csv.writer(loss_csv)
    loss_writer.writerow(['epoch', 'total_loss', 'box_loss', 'obj_loss', 'cls_loss'])

    logger.info(f"开始训练，共 {args.epochs} 个 epoch")
    logger.info(('%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))

    # ------------------- 训练循环 -------------------
    for epoch in range(args.epochs):
        model.train()
        mloss = torch.zeros(3, device=device)  # [box, obj, cls]

        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for i, (imgs, targets) in pbar:
            imgs = imgs.to(device, non_blocking=True)

            try:
                num_labels = sum(len(t) for t in targets)
            except Exception:
                num_labels = 0

            outputs = model(imgs)
            loss_dict = criterion(outputs, targets, epoch)
            loss = loss_dict['losses']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_items = torch.stack([
                loss_dict['loss_box'].detach(),
                loss_dict['loss_obj'].detach(),
                loss_dict['loss_cls'].detach(),
            ])
            mloss = (mloss * i + loss_items) / (i + 1)

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            s = ('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch + 1}/{args.epochs}', mem,
                mloss[0], mloss[1], mloss[2],
                num_labels, imgs.shape[-1]
            )
            pbar.set_description(s)

            if i == len(dataloader) - 1:
                avg_box, avg_obj, avg_cls = mloss.tolist()
                avg_loss = avg_box + avg_obj + avg_cls

        # ---- Epoch 结束 ----
        epoch_list.append(epoch + 1)
        loss_total_list.append(avg_loss)
        loss_obj_list.append(avg_obj)
        loss_cls_list.append(avg_cls)
        loss_box_list.append(avg_box)

        # 每个 epoch 实时写入 loss CSV
        loss_writer.writerow([epoch + 1, f'{avg_loss:.6f}', f'{avg_box:.6f}',
                              f'{avg_obj:.6f}', f'{avg_cls:.6f}'])
        loss_csv.flush()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "best_yolov3_mcu.pth"))

    logger.info(f"训练完成！最佳模型在 epoch {best_epoch}，loss={best_loss:.4f}")
    loss_csv.close()
    logger.info(f"Loss 历史已保存至 {loss_csv_path}")

    # ------------------- 加载最优权重评估 -------------------
    best_ckpt = os.path.join(args.save_dir, "best_yolov3_mcu.pth")
    if os.path.exists(best_ckpt):
        logger.info("加载最佳权重进行评估...")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    evaluate_and_plot(model, dataloader, device, num_classes,
                      args.save_dir, class_names=class_names_map)

    # ------------------- 绘图 -------------------
    plots_data = [
        ('Total Loss', loss_total_list),
        ('Objectness Loss', loss_obj_list),
        ('Classification Loss', loss_cls_list),
        ('Box Loss', loss_box_list),
    ]

    plt.style.use('default')
    plt.rc('font', size=10)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    ax = ax.ravel()

    for idx, (title, y_data) in enumerate(plots_data):
        ax[idx].plot(epoch_list, y_data, marker='.', label='train',
                     linewidth=2, markersize=8, color='C0')
        ax[idx].set_title(title, fontsize=12)
        ax[idx].set_xlabel('Epoch')
        ax[idx].grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(args.save_dir, "results_yolo_style.png")
    logger.info(f"保存训练曲线到 {save_path}")
    fig.savefig(save_path, dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
