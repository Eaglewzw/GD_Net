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
    def __init__(self, img_dir, label_dir, img_size=640, transform=None, class_mapping=None):

        self.img_dir = img_dir  # 添加这行
        self.label_dir = label_dir  # 添加这行
        # 支持 jpg/jpeg/png
        self.img_files = sorted(
            glob.glob(os.path.join(img_dir, '*.jpg')) +
            glob.glob(os.path.join(img_dir, '*.jpeg')) +
            glob.glob(os.path.join(img_dir, '*.png'))
        )
        # 修改：查找XML标签文件
        self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.xml')))

        # 设置类别映射，如果没有提供则使用默认的
        if class_mapping is None:
            self.class_mapping = {'drone': 0, 'Drone': 0, 'DRONE': 0, 'droned': 0}  # 添加不同大小写形式
        else:
            self.class_mapping = class_mapping

        # 确保文件数量匹配
        if len(self.img_files) != len(self.label_files):
            print(f"警告: 图像数量({len(self.img_files)})与标签数量({len(self.label_files)})不匹配")
            # 验证文件名是否匹配（基于文件名基本部分）
            img_basenames = {os.path.splitext(os.path.basename(f))[0] for f in self.img_files}
            label_basenames = {os.path.splitext(os.path.basename(f))[0] for f in self.label_files}
            unmatched = img_basenames.symmetric_difference(label_basenames)
            if unmatched:
                print(f"警告: 发现未匹配的文件: {unmatched}")

        self.img_size = img_size
        self.transform = transform

        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}. "
                                    f"Supported formats: jpg, jpeg, png")
        if len(self.label_files) == 0:
            raise FileNotFoundError(f"No labels found in {label_dir}. "
                                    f"Expected .xml files for PASCAL VOC format")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size

        # 构造对应的 XML 路径
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, img_name + '.xml')

        boxes = []
        labels = []

        if os.path.exists(label_path):
            tree = ET.parse(label_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text.strip()        # 这里就是 "Drone"
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
