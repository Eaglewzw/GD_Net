"""
prepare_udacity.py
------------------
将 UdacitySelfDrivingCar 数据集（VOC XML 格式）整理为当前模型可直接训练的结构：

  Driving_Car/
  ├── Annotations/        (原始 XML，不动)
  ├── JPEGImages/         (原始图片，不动)
  └── ImageSets/
      └── Main/
          ├── train.txt   <- 80%
          ├── val.txt     <- 10%
          └── test.txt    <- 10%

同时打印：
  - 数据集类别统计
  - 空标注（无目标）图片数量
  - 各划分样本数

用法：
  python prepare_udacity.py
  python prepare_udacity.py --root /your/path/Driving_Car --split 0.8 0.1 0.1 --seed 42
"""

import os
import random
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


# ── 数据集 11 类 → 合并为 5 类（适配轻量模型，可按需修改）──
# 若想训练全部 11 类，将 CLASS_MAPPING 改为每类单独映射即可
CLASS_MAPPING = {
    'car':                    'car',
    'truck':                  'truck',
    'pedestrian':             'pedestrian',
    'biker':                  'biker',
    'trafficLight':           'trafficLight',
    'trafficLight-Red':       'trafficLight',
    'trafficLight-Green':     'trafficLight',
    'trafficLight-Yellow':    'trafficLight',
    'trafficLight-RedLeft':   'trafficLight',
    'trafficLight-GreenLeft': 'trafficLight',
    'trafficLight-YellowLeft':'trafficLight',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/media/verser/robot/Dataset/UdacitySelfDrivingCarDataset/Driving_Car',
                        help='数据集根目录（含 Annotations/ 和 JPEGImages/）')
    parser.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='train/val/test 比例，默认 0.8 0.1 0.1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-empty', action='store_true',
                        help='跳过没有任何目标的图片（不加入训练集）')
    return parser.parse_args()


def scan_annotations(ann_dir: Path):
    """扫描所有 XML，返回 (有效id列表, 类别计数, 空标注id列表)"""
    class_counter = Counter()
    empty_ids = []
    valid_ids = []

    xml_files = sorted(ann_dir.glob('*.xml'))
    if not xml_files:
        raise FileNotFoundError(f"Annotations 目录下没有 XML 文件: {ann_dir}")

    for xml_path in xml_files:
        stem = xml_path.stem          # e.g. "00001"
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"[WARN] 解析失败，跳过: {xml_path.name} ({e})")
            continue

        objects = root.findall('object')
        if not objects:
            empty_ids.append(stem)
        else:
            for obj in objects:
                name = obj.find('name').text.strip()
                mapped = CLASS_MAPPING.get(name, name)
                class_counter[mapped] += 1
            valid_ids.append(stem)

    return valid_ids, class_counter, empty_ids


def split_ids(ids, ratios, seed):
    """按比例随机划分 id 列表，返回 (train, val, test)"""
    random.seed(seed)
    ids = ids.copy()
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    train = ids[:n_train]
    val   = ids[n_train:n_train + n_val]
    test  = ids[n_train + n_val:]
    return train, val, test


def write_split(out_dir: Path, split_name: str, ids: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{split_name}.txt'
    with open(out_path, 'w') as f:
        f.write('\n'.join(ids) + '\n')
    print(f"  {split_name:6s}: {len(ids):6d} 张  →  {out_path}")


def print_class_info(class_counter, mapping):
    """打印类别映射和样本统计"""
    print("\n── 类别统计（合并后）──")
    total = sum(class_counter.values())
    for cls, cnt in sorted(class_counter.items(), key=lambda x: -x[1]):
        print(f"  {cls:<25s}  {cnt:>7d}  ({cnt/total*100:.1f}%)")
    print(f"  {'合计':<25s}  {total:>7d}")

    unique_mapped = sorted(set(mapping.values()))
    print(f"\n── 合并后共 {len(unique_mapped)} 个类别 ──")
    for c in unique_mapped:
        originals = [k for k, v in mapping.items() if v == c]
        print(f"  {c:<20s} ← {', '.join(originals)}")


def main():
    args = parse_args()
    root = Path(args.root)
    ann_dir = root / 'Annotations'
    img_dir = root / 'JPEGImages'
    out_dir = root / 'ImageSets' / 'Main'

    # 检查路径
    if not ann_dir.exists():
        raise FileNotFoundError(f"找不到 Annotations 目录: {ann_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"找不到 JPEGImages 目录: {img_dir}")

    print(f"数据集路径: {root}")
    print("扫描 XML 标注中...")

    valid_ids, class_counter, empty_ids = scan_annotations(ann_dir)

    print(f"\n── 文件统计 ──")
    print(f"  总图片数:      {len(list(img_dir.glob('*.jpg'))):>6d}")
    print(f"  有目标的图片:  {len(valid_ids):>6d}")
    print(f"  无目标的图片:  {len(empty_ids):>6d}")

    print_class_info(class_counter, CLASS_MAPPING)

    # 决定参与训练的 id
    train_pool = valid_ids
    if not args.skip_empty:
        train_pool = valid_ids + empty_ids
        print(f"\n[INFO] 空标注图片也加入划分（共 {len(train_pool)} 张）")
        print("[INFO] 如需过滤空标注，使用 --skip-empty")
    else:
        print(f"\n[INFO] 已跳过 {len(empty_ids)} 张无目标图片")

    # 划分
    ratios = args.split
    assert abs(sum(ratios) - 1.0) < 1e-6, "split 比例之和必须为 1"
    train_ids, val_ids, test_ids = split_ids(train_pool, ratios, args.seed)

    print(f"\n── 数据集划分（seed={args.seed}）──")
    write_split(out_dir, 'train', train_ids)
    write_split(out_dir, 'val',   val_ids)
    write_split(out_dir, 'test',  test_ids)

    print(f"\n完成！ImageSets 已写入: {out_dir}")
    print("\n── 训练命令示例 ──")
    print(f"python train.py \\")
    print(f"  --data-root {root} \\")
    print(f"  --num-classes {len(set(CLASS_MAPPING.values()))} \\")
    print(f"  --img-size 256 \\")
    print(f"  --epochs 100 \\")
    print(f"  --batch-size 16")


if __name__ == '__main__':
    main()
