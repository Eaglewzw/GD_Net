import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms
import xml.etree.ElementTree as ET


class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, transform=None,
                 class_mapping=None, label_format='voc'):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label_format = label_format

        # 支持 jpg/jpeg/png
        self.img_files = sorted(
            glob.glob(os.path.join(img_dir, '*.jpg')) +
            glob.glob(os.path.join(img_dir, '*.jpeg')) +
            glob.glob(os.path.join(img_dir, '*.png'))
        )

        if label_format == 'yolo':
            self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
            self.valid_class_ids = set(class_mapping.values()) if class_mapping else None
        else:
            self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.xml')))

        # 设置类别映射
        if class_mapping is None:
            self.class_mapping = {'drone': 0, 'Drone': 0, 'DRONE': 0, 'droned': 0}
        else:
            self.class_mapping = class_mapping

        # 文件数量统计（YOLO格式允许图片无标签，只作信息提示）
        img_set = {os.path.splitext(os.path.basename(f))[0] for f in self.img_files}
        lbl_set = {os.path.splitext(os.path.basename(f))[0] for f in self.label_files}
        imgs_no_label = img_set - lbl_set
        labels_no_img = lbl_set - img_set
        if imgs_no_label:
            print(f"信息: {len(imgs_no_label)} 张图片没有对应的标签文件")
        if labels_no_img:
            print(f"信息: {len(labels_no_img)} 个标签文件没有对应的图片")

        self.img_size = img_size
        self.transform = transform

        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}. "
                                    f"Supported formats: jpg, jpeg, png")
        if len(self.label_files) == 0:
            ext = 'txt' if label_format == 'yolo' else 'xml'
            raise FileNotFoundError(f"No labels found in {label_dir}. "
                                    f"Expected .{ext} files for {label_format.upper()} format")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = '.txt' if self.label_format == 'yolo' else '.xml'
        label_path = os.path.join(self.label_dir, img_name + ext)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            if self.label_format == 'yolo':
                boxes, labels = self._parse_yolo_label(label_path, orig_w, orig_h)
            else:
                boxes, labels = self._parse_voc_label(label_path)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,))

        # Letterbox 变换
        img_resized, ratio, (pad_w, pad_h) = self.letterbox(img_pil, self.img_size)

        # 坐标同步变换
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= ratio      # x1, x2
            boxes[:, [1, 3]] *= ratio      # y1, y2
            boxes[:, [0, 2]] += pad_w      # 左右 padding
            boxes[:, [1, 3]] += pad_h      # 上下 padding

        img_tensor = self.transform(img_resized)

        target = {
            'boxes': boxes,      # xyxy 格式，已在 640×640 范围内
            'labels': labels,
        }

        return img_tensor, target


    def _parse_yolo_label(self, label_path, img_w, img_h):
        """解析 YOLO 格式标签: class_id cx cy w h (归一化坐标) → xyxy"""
        boxes, labels = [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                if w <= 0 or h <= 0:
                    continue
                if self.valid_class_ids is not None and cls_id not in self.valid_class_ids:
                    print(f"[Warning] Unknown class '{cls_id}' in {label_path}")
                    continue
                # 归一化 cx,cy,w,h → 绝对 xmin,ymin,xmax,ymax
                xmin = (cx - w / 2) * img_w
                ymin = (cy - h / 2) * img_h
                xmax = (cx + w / 2) * img_w
                ymax = (cy + h / 2) * img_h
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)
        return boxes, labels

    def _parse_voc_label(self, label_path):
        """解析 VOC XML 标签"""
        boxes, labels = [], []
        tree = ET.parse(label_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            if name not in self.class_mapping:
                print(f"[Warning] Unknown class '{name}' in {label_path}")
                continue
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            if xmin >= xmax or ymin >= ymax:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_mapping[name])
        return boxes, labels

    def letterbox(self, img_pil, new_shape=640, color=(114, 114, 114)):
        img = np.array(img_pil)
        h, w = img.shape[:2]
        ratio = min(new_shape / w, new_shape / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        if (w, h) != (new_w, new_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = (new_shape - new_w) // 2
        dh = (new_shape - new_h) // 2

        img = cv2.copyMakeBorder(img, dh, new_shape - new_h - dh,
                                 dw, new_shape - new_w - dw,
                                 cv2.BORDER_CONSTANT, value=color)
        return Image.fromarray(img.astype(np.uint8)), ratio, (dw, dh)
