import os
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


def load_data_cfg(yaml_path):
    """从 yaml 文件读取数据集配置，返回 (data_root, num_classes, class_mapping)"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data_root     = data['path']
    class_mapping = data['class_mapping']
    num_classes   = len(set(class_mapping.values()))
    class_names   = data.get('names', {})

    logger.info(f"数据集配置: {yaml_path}")
    logger.info(f"  路径: {data_root}")
    logger.info(f"  类别数: {num_classes}  {list(class_names.values())}")
    return data_root, num_classes, class_mapping


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv3-McuNet Training")
    # 数据集配置 yaml（优先级高于 --data-root / --num-classes）
    parser.add_argument('--data', type=str, default='/home/verser/Python/GD_Net/data/udacity.yaml',
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
                        default='',
                        help='预训练权重路径，留空则从头训练')
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--img-size', type=int, default=160)
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
        data_root, num_classes, class_mapping = load_data_cfg(args.data)
    else:
        data_root    = args.data_root
        num_classes  = args.num_classes
        class_mapping = None   # 使用 YoloDataset 内置默认值

    img_dir   = os.path.join(data_root, 'JPEGImages')
    label_dir = os.path.join(data_root, 'Annotations')

    dataset = YoloDataset(img_dir=img_dir, label_dir=label_dir,
                          img_size=args.img_size,
                          transform=torchvision.transforms.ToTensor(),
                          class_mapping=class_mapping)
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "best_yolov3_mcu.pth"))

    logger.info(f"训练完成！最佳模型在 epoch {best_epoch}，loss={best_loss:.4f}")

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
