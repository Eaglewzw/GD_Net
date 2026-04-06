import cv2
import random
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
import torch.utils.data as data
import yaml

try:
    from .gd_net.augment_strong import MosaicAugment, MixupAugment
except:
    from  gd_net.augment_strong import MosaicAugment, MixupAugment


def load_data_cfg(yaml_path):
    """从 yaml 文件读取数据集配置，返回 (data_root, class_names, class_mapping)"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    data_root     = data['path']
    names         = data.get('names', {})
    class_mapping = data.get('class_mapping', {})
    # names 是 {id: name} 字典，转为按 id 排序的 tuple
    class_names = tuple(names[i] for i in sorted(names.keys()))
    return data_root, class_names, class_mapping


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Arguments:
        class_to_ind (dict): classname -> index 映射，来自 yaml class_mapping
        keep_difficult (bool): 是否保留 difficult 样本
    """

    def __init__(self, class_to_ind, keep_difficult=False):
        self.class_to_ind = class_to_ind
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult_node = obj.find('difficult')
            difficult = int(difficult_node.text) == 1 if difficult_node is not None else False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.strip()
            if name not in self.class_to_ind:
                print(f"[Warning] Unknown class '{name}', skipped.")
                continue
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDataset(data.Dataset):
    def __init__(self,
                 img_size     :int = 640,
                 data_dir     :str = None,
                 data         :str = None,
                 image_sets   = ['trainval', 'train'],
                 trans_config = None,
                 transform    = None,
                 is_train     :bool = False,
                 ):
        # ----------- Basic parameters -----------
        self.img_size = img_size
        self.image_set = image_sets
        self.is_train = is_train

        # ----------- 数据集配置：优先从 yaml 读取 -----------
        if data is not None:
            data_root, class_names, class_mapping = load_data_cfg(data)
            print(f"[VOCDataset] 加载配置: {data}")
            print(f"[VOCDataset] 数据路径: {data_root}")
            print(f"[VOCDataset] 类别 ({len(class_names)}): {class_names}")
        else:
            # 兼容旧用法：data_dir 直接传入，类别沿用固定的 drone
            data_root   = data_dir
            class_names = ('drone',)
            class_mapping = {'drone': 0}
            print(f"[VOCDataset] 未指定 yaml，使用默认类别: {class_names}")

        self.class_names  = class_names
        self.class_mapping = class_mapping
        self.target_transform = VOCAnnotationTransform(class_to_ind=class_mapping)

        # ----------- Path parameters -----------
        self.root = data_root
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        # ----------- Data parameters -----------
        self.ids = list()
        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))


        for name in image_sets:  # 只遍历数据集划分名称
            rootpath = self.root  # 直接使用根目录
            txt_path = osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')
            if not osp.exists(txt_path):
                print(f"[VOCDataset] 跳过不存在的划分文件: {txt_path}")
                continue
            for line in open(txt_path):
                self.ids.append((rootpath, line.strip()))
        self.dataset_size = len(self.ids)

        # ----------- Transform parameters -----------
        self.trans_config = trans_config
        self.transform = transform
        # ----------- Strong augmentation -----------
        if is_train:
            self.mosaic_prob = trans_config['mosaic_prob'] if trans_config else 0.0
            self.mixup_prob  = trans_config['mixup_prob']  if trans_config else 0.0
            self.mosaic_augment = MosaicAugment(img_size, trans_config, is_train) if self.mosaic_prob > 0. else None
            self.mixup_augment  = MixupAugment(img_size, trans_config)            if self.mixup_prob > 0.  else None
        else:
            self.mosaic_prob = 0.0
            self.mixup_prob  = 0.0
            self.mosaic_augment = None
            self.mixup_augment  = None
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))

    # ------------ Basic dataset function ------------
    def __getitem__(self, index):
        image, target, deltas = self.pull_item(index)
        return image, target, deltas

    def __len__(self):
        return self.dataset_size

    # ------------ Mosaic & Mixup ------------
    def load_mosaic(self, index):
        # ------------ Prepare 4 indexes of images ------------
        ## Load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]

        ## Load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # ------------ Mosaic augmentation ------------
        image, target = self.mosaic_augment(image_list, target_list)

        return image, target

    def load_mixup(self, origin_image, origin_target):
        # ------------ Load a new image & target ------------
        if self.mixup_augment.mixup_type == 'yolov5':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_mosaic(new_index)
        elif self.mixup_augment.mixup_type == 'yolox':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_image_target(new_index)

        # ------------ Mixup augmentation ------------
        image, target = self.mixup_augment(origin_image, origin_target, new_image, new_target)

        return image, target

    # ------------ Load data function ------------
    def load_image_target(self, index):
        # load an image
        image, _ = self.pull_image(index)
        height, width, channels = image.shape

        # laod an annotation
        anno, _ = self.pull_anno(index)

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [height, width]
        }

        return image, target

    def pull_item(self, index):
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)

        # MixUp
        if random.random() < self.mixup_prob:
            image, target = self.load_mixup(image, target)

        # augment
        image, target, deltas = self.transform(image, target, mosaic)

        return image, target, deltas

    def pull_image(self, index):
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        return image, img_id

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self.target_transform(anno)

        return anno, img_id


if __name__ == "__main__":
    import time
    import argparse
    from factory import build_transform

    parser = argparse.ArgumentParser(description='VOC-Dataset')

    # opt
    parser.add_argument('--data', default='../data/standford.yaml',
                        help='数据集配置 yaml（data/目录下）')
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='input image size.')
    parser.add_argument('--aug_type', type=str, default='ssd',
                        help='augmentation type: ssd, yolo.')
    parser.add_argument('--mosaic', default=0., type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=0., type=float,
                        help='mixup augmentation.')
    parser.add_argument('--mixup_type', type=str, default='yolov5_mixup',
                        help='mixup augmentation.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')

    args = parser.parse_args()

    trans_config = {
        'aug_type': args.aug_type,    # optional: ssd, yolov5
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std':  [58.395, 57.12, 57.375],
        'use_ablu': True,
        # Basic Augment
        'affine_params': {
            'degrees': 0.0,
            'translate': 0.2,
            'scale': [0.1, 2.0],
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        },
        # Mosaic & Mixup
        'mosaic_keep_ratio': False,
        'mosaic_prob': args.mosaic,
        'mixup_prob':  args.mixup,
        'mosaic_type': 'yolov5',
        'mixup_type':  'yolov5',
        'mixup_scale': [0.5, 1.5]
    }
    transform, trans_cfg = build_transform(args, trans_config, 32, args.is_train)
    pixel_mean = transform.pixel_mean
    pixel_std  = transform.pixel_std
    color_format = transform.color_format

    dataset = VOCDataset(
        img_size=args.img_size,
        data=args.data,
        image_sets=['trainval', 'train'],
        trans_config=trans_config,
        transform=transform,
        is_train=args.is_train,
    )

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(len(dataset.class_names))]
    print('Data length: ', len(dataset))

    for i in range(1000):
        t0 = time.time()
        image, target, deltas = dataset.pull_item(i)
        print("Load data: {} s".format(time.time() - t0))

        # to numpy
        image = image.permute(1, 2, 0).numpy()

        # denormalize
        image = image * pixel_std + pixel_mean
        if color_format == 'rgb':
            # RGB to BGR
            image = image[..., (2, 1, 0)]

        # to uint8
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            if x2 - x1 > 1 and y2 - y1 > 1:
                cls_id = int(label)
                color = class_colors[cls_id]
                cls_name = dataset.class_names[cls_id]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image, cls_name, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        cv2.waitKey(0)