import os
import torch
import torch.optim as optim
import torchvision
from gd_net.yolov3_mcu_net import YOLOv3_McuNet
from gd_net.yolo_dataset_loader import YoloDataset, DataLoader
from gd_net.loss import build_criterion
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import gc

# ==================== 修复 meshgrid 警告 ====================
torch_meshgrid_original = torch.meshgrid
def fixed_meshgrid(*tensors, **kwargs):
    kwargs.setdefault('indexing', 'ij')
    return torch_meshgrid_original(*tensors, **kwargs)
torch.meshgrid = fixed_meshgrid
# ===========================================================

# 关闭 cudnn benchmark（避免 CUDNN_STATUS_NOT_SUPPORTED）
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets

# ------------------- 配置 -------------------
torch.cuda.set_device(0)  # 设置默认GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 32
img_size = 256
num_classes = 1

cfg = {
    'loss_obj_weight': 1.0,  # 提高obj权重
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    'obj_pos_weight': 30.0,   # 新增：obj正样本权重
    'iou_thresh': 0.25,       # 降低IoU阈值，便于小目标分配
    'conf_thresh': 0.25,
    'nms_thresh': 0.25,
    'pretrained': True,
    'stride': [8, 16, 32],
    'width': 1.0,
    'depth': 1.0,
    'max_stride': 32,
    'neck': 'sppf',
    'neck_act': 'silu',
    'neck_norm': 'BN',
    'neck_depthwise': False,
    'expand_ratio': 0.5,
    'pooling_size': 5,
    'fpn': 'yolov3_fpn',
    'fpn_act': 'silu',
    'fpn_norm': 'BN',
    'fpn_depthwise': False,
    'head': 'decoupled_head',
    'head_act': 'silu',
    'head_norm': 'BN',
    'num_cls_head': 2,
    'num_reg_head': 2,
    'head_depthwise': False,

    'anchor_size': [[16, 21], [18, 24], [21, 21],     # P3 (小目标)
                    [22, 26], [24, 31], [26, 22],     # P4 (中目标)
                    [30, 28], [36, 33], [40, 43]],    # P5 (大目标)

    # 'anchor_size': [[8, 10], [24, 26], [41, 48],            # P3
    #                 [70, 33], [55, 74], [67, 100],          # P4
    #                 [97, 69], [84, 122], [153, 153]],       # P5
}

# ------------------- 数据集 -------------------
dataset_path = '/media/verser/robot/Dataset/DT_Drone'
img_dir   = os.path.join(dataset_path, 'JPEGImages')
label_dir = os.path.join(dataset_path, 'Annotations')


# dataset_path = '/media/verse/roboot1/dataset/VisDrone'
# img_dir   = os.path.join(dataset_path, 'images')
# label_dir = os.path.join(dataset_path, 'Annotations')

dataset = YoloDataset(img_dir=img_dir, label_dir=label_dir,
                      img_size=img_size, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, persistent_workers=False, pin_memory=False)

# ------------------- 模型 / 损失 / 优化器 -------------------
model = YOLOv3_McuNet(cfg, device, num_classes=num_classes, trainable=True).to(device)

pretrained_path = "/home/verser/Python/GD_Net/checkpoints/best_yolov3_mcu.pth"
if cfg['pretrained'] and os.path.exists(pretrained_path):
    print(f"🔄 正在加载预训练模型: {pretrained_path} ...")
    try:
        # 1. 加载 checkpoint
        checkpoint = torch.load(pretrained_path, map_location=device)

        # 2. 判断 checkpoint 格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 3. 加载权重到模型
        model.load_state_dict(state_dict, strict=False)
        print("✅ 预训练权重加载成功！")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("⚠️ 将从头开始训练 (主干网络可能仍会加载其自己的预训练权重)")
else:
    print("ℹ️ 未指定预训练模型或文件不存在，将从头开始训练。")


# ==================================================================
criterion = build_criterion(cfg, device, num_classes=num_classes)
optimizer = optim.SGD(model.parameters(),
                      lr=0.012,
                      momentum=0.937,
                      weight_decay=5e-4,
                      nesterov=True)

# ------------------- 添加 OneCycleLR -------------------
total_steps = len(dataloader) * epochs
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.012,
    total_steps=total_steps,
    pct_start=0.1,
    div_factor=300,
    final_div_factor=1e5,
    anneal_strategy='cos'
)

# ------------------- 训练准备 -------------------
best_loss, best_epoch = float('inf'), 0
save_dir = "/home/verser/Python/GD_Net/checkpoints"
os.makedirs(save_dir, exist_ok=True)

epoch_list = []; loss_total_list = []; loss_obj_list = []; loss_cls_list = []; loss_box_list = []
global_start = time.time()

# ------------------- 训练循环 -------------------
for epoch in range(epochs):
    # 只需要在 epoch 开始时清理一次即可，甚至可以不需要
    model.train()
    
    # 初始化统计变量 (纯数值)
    total_loss_epoch = 0.0
    total_obj_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0
    count = 0

    # 使用 with 语句管理 tqdm，确保资源释放
    pbar = tqdm(dataloader,
                desc=f"Epoch {epoch+1:>3}/{epochs}",
                leave=False, ncols=130,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    
    epoch_start = time.time()
    
    for imgs, targets in pbar:
        # 数据迁移
        imgs = imgs.to(device, non_blocking=True)

        # 前向传播
        outputs = model(imgs)
        loss_dict = criterion(outputs, targets, epoch)
        loss = loss_dict['losses']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- 使用 .item() 避免计算图累积 ---
        total_loss_epoch += loss.item()
        total_obj_loss   += float(loss_dict['loss_obj']) 
        total_cls_loss   += float(loss_dict['loss_cls'])
        total_box_loss   += float(loss_dict['loss_box'])
        
        count += 1

        cur_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            "loss": f"{total_loss_epoch/count:.4f}",
            "obj" : f"{total_obj_loss/count:.4f}",
            "cls" : f"{total_cls_loss/count:.4f}",
            "box" : f"{total_box_loss/count:.4f}",
            "lr"  : f"{cur_lr:.2e}"
        })
        

    # ---- 本 epoch 统计 ----
    avg_loss = total_loss_epoch / count
    avg_obj  = total_obj_loss   / count
    avg_cls  = total_cls_loss   / count
    avg_box  = total_box_loss   / count

    epoch_list.append(epoch + 1)
    loss_total_list.append(avg_loss)
    loss_obj_list.append(avg_obj)
    loss_cls_list.append(avg_cls)
    loss_box_list.append(avg_box)

    epoch_time = time.time() - epoch_start
    total_time = time.time() - global_start
    eta = (epochs - epoch - 1) * (total_time / (epoch + 1)) if epoch > 0 else 0

    print(f"\nEpoch {epoch+1}/{epochs} | "
          f"total:{avg_loss:.4f} | obj:{avg_obj:.4f} | cls:{avg_cls:.4f} | box:{avg_box:.4f} | "
          f"time:{epoch_time:.1f}s | ETA:{eta/60:.1f}min | lr:{cur_lr:.2e}")
    print("=" * 80)

    if avg_loss < best_loss:
        best_loss, best_epoch = avg_loss, epoch + 1
        save_path = os.path.join(save_dir, "best_yolov3_mcu.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ [Best Model Updated] Epoch {best_epoch}, loss={best_loss:.4f} → {save_path}")
    
    # 每个 epoch 结束时清理一次显存碎片（可选，通常不需要，除非显存非常紧张）
    # torch.cuda.empty_cache()


print(f"\nTraining finished! Best model at epoch {best_epoch}, loss={best_loss:.4f}")


# ------------------- 绘图 -------------------
# float 类型
epoch_list      = [float(x) for x in epoch_list]
loss_total_list = [float(x) for x in loss_total_list]
loss_obj_list   = [float(x) for x in loss_obj_list]
loss_cls_list   = [float(x) for x in loss_cls_list]
loss_box_list   = [float(x) for x in loss_box_list]


# 设置风格
plt.style.use('seaborn-v0_8')
figsize_sub = (8, 5.5)
font_size_title = 16
font_size_label = 14
dpi = 300

# 创建 2×2 子图
fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=dpi)
axes = axes.ravel()

# 1. Total Loss
axes[0].plot(epoch_list, loss_total_list, color='black', linewidth=2.8, marker='o', markersize=4)
axes[0].set_title('Total Loss', fontsize=font_size_title, fontweight='bold', pad=15)
axes[0].set_xlabel('Epoch', fontsize=font_size_label)
axes[0].set_ylabel('Total Loss', fontsize=font_size_label)
axes[0].grid(True, linestyle='--', alpha=0.7)

# 2. Objectness Loss
axes[1].plot(epoch_list, loss_obj_list, color='#1f77b4', linewidth=2.8, marker='s', markersize=4)
axes[1].set_title('Objectness Loss', fontsize=font_size_title, fontweight='bold', pad=15)
axes[1].set_xlabel('Epoch', fontsize=font_size_label)
axes[1].set_ylabel('Objectness Loss', fontsize=font_size_label)
axes[1].grid(True, linestyle='--', alpha=0.7)

# 3. Classification Loss
axes[2].plot(epoch_list, loss_cls_list, color='#2ca02c', linewidth=2.8, marker='^', markersize=4)
axes[2].set_title('Classification Loss', fontsize=font_size_title, fontweight='bold', pad=15)
axes[2].set_xlabel('Epoch', fontsize=font_size_label)
axes[2].set_ylabel('Classification Loss', fontsize=font_size_label)
axes[2].grid(True, linestyle='--', alpha=0.7)

# 4. Box Regression Loss
axes[3].plot(epoch_list, loss_box_list, color='#d62728', linewidth=2.8, marker='D', markersize=4)
axes[3].set_title('Box Regression Loss', fontsize=font_size_title, fontweight='bold', pad=15)
axes[3].set_xlabel('Epoch', fontsize=font_size_label)
axes[3].set_ylabel('Box Loss', fontsize=font_size_label)
axes[3].grid(True, linestyle='--', alpha=0.7)

# 整体布局优化
plt.tight_layout(pad=3.0)
plt.savefig("training_all_losses.png", dpi=dpi, bbox_inches='tight')
plt.show()