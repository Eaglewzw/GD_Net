import torch
import cv2
import numpy as np
import time
import os
import argparse
import logging
import yaml
from pathlib import Path

from gd_net.model import YOLOv3_McuNet
from cfg import cfg
from utils.plotting import visualize
from utils.augmentations import letterbox

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class FPSCounter:
    """ 滑动窗口 FPS 计数器 """

    def __init__(self, window_size=60):
        self.window_size = window_size
        self.timestamps = []

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        if len(self.timestamps) >= 2:
            return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])
        return 0.0


# ==================== 核心推理引擎类 ====================

class MCUNetDetector:
    def __init__(self, weights_path, data_cfg='', img_size=256, device=None):
        self.img_size = img_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── 从 yaml 读取类别信息 ──
        if data_cfg and os.path.exists(data_cfg):
            with open(data_cfg, 'r') as f:
                data = yaml.safe_load(f)
            names_dict     = data.get('names', {0: 'object'})
            self.class_names  = [names_dict[i] for i in sorted(names_dict)]
            num_classes    = len(self.class_names)
            self.num_classes = num_classes
        else:
            self.class_names  = ['Car']
            num_classes    = 1

        # 每个类别分配一个颜色
        palette = [
            (0, 255, 255), (0, 128, 255), (0, 255, 128),
            (255, 128, 0), (255, 0, 128), (128, 0, 255),
        ]
        self.class_colors = [palette[i % len(palette)] for i in range(num_classes)]

        # 加载模型
        logger.info(f"Loading model from {weights_path} to {self.device}")
        logger.info(f"Classes ({num_classes}): {self.class_names}")
        self.model = YOLOv3_McuNet(
            cfg, device=self.device, num_classes=num_classes, trainable=False,
            conf_thresh=cfg['conf_thresh'],
            nms_thresh=cfg['nms_thresh'],
        )

        ckpt = torch.load(weights_path, map_location='cpu')
        # 兼容 state_dict 可能包含 'model' 键的情况
        if 'model' in ckpt:
            ckpt = ckpt['model']

        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 预热模型 (Warmup) - 避免第一次推理时间过长
        logger.info("Warming up model...")
        dummy_input = torch.zeros(1, 3, img_size, img_size).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)
        logger.info("Model loaded and ready.")

    def preprocess(self, img_raw):
        """ 图像预处理 """
        t_start = time.time()
        x, ratio, (dw, dh) = letterbox(img_raw, new_shape=self.img_size)
        x = x.to(self.device)
        t_end = time.time()
        return x, ratio, (dw, dh), (t_end - t_start) * 1000

    def infer_single_frame(self, frame):
        """
        处理单帧的核心逻辑
        返回: 绘制好的图像, 结果信息(用于打印), 时间统计
        """
        # 1. 预处理
        x, ratio, (dw, dh), t_pre = self.preprocess(frame)

        # 2. 推理
        t_infer_start = time.time()
        with torch.no_grad():
            outputs = self.model(x)
        t_infer_end = time.time()
        t_infer = (t_infer_end - t_infer_start) * 1000

        # 3. 后处理坐标还原
        bboxes = outputs['bboxes']
        scores = outputs['scores']
        labels = outputs['labels']

        if len(bboxes) > 0:
            bboxes = np.array(bboxes).reshape(-1, 4)
            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / ratio
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / ratio

        # 4. 可视化绘制
        vis_frame = visualize(frame.copy(), bboxes, scores, labels.astype(int),
                              self.class_colors, self.class_names, list(range(self.num_classes)))

        # return vis_frame, t_pre, t_infer, len(bboxes)

        return vis_frame, t_pre, t_infer, len(bboxes), labels

    def run(self, input_path, save_dir='./det_results/'):
        """ 主入口：自动判断是图片还是视频 """
        input_path = Path(input_path)
        os.makedirs(save_dir, exist_ok=True)

        # 定义支持的格式
        IMG_FORMATS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo'}
        VID_FORMATS = {'.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv'}

        file_ext = input_path.suffix.lower()

        if file_ext in IMG_FORMATS:
            self._process_image(input_path, save_dir)
        elif file_ext in VID_FORMATS:
            self._process_video(input_path, save_dir)
        else:
            logger.warning(f"Unsupported file format: {file_ext}")

    def _process_image(self, img_path, save_dir):
        logger.info(f"Processing Image: {img_path}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.error("Failed to load image.")
            return

        # 执行推理
        # vis_frame, t_pre, t_infer, count = self.infer_single_frame(frame)
        vis_frame, t_pre, t_infer, count, labels = self.infer_single_frame(frame)

        # 绘制文本信息 (不需要FPS)
        info_text = f"Car: {count} | Infer: {t_infer:.1f}ms"
        cv2.putText(vis_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # 保存
        save_name = os.path.join(save_dir, img_path.name)
        cv2.imwrite(save_name, vis_frame)
        logger.info(f"Saved to: {save_name} | {info_text}")

    def _process_video(self, vid_path, save_dir):
        logger.info(f"Processing Video: {vid_path}")
        cap = cv2.VideoCapture(str(vid_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        save_name = os.path.join(save_dir, vid_path.stem + '_det.mp4')
        # mp4v 兼容性较好
        out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        fps_counter = FPSCounter()
        frame_id = 0

        # 临时窗口 (可选)
        cv2.namedWindow('YOLOv3-MCU', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLOv3-MCU', 1280, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # 推理
            # vis_frame, t_pre, t_infer, count = self.infer_single_frame(frame)
            vis_frame, t_pre, t_infer, count, labels = self.infer_single_frame(frame)

            # 计算平滑 FPS
            fps_real = fps_counter.tick()

            # 绘制详细信息 (视频需要FPS)
            y_offset = 40
            texts = [
                f"FPS: {fps_real:.1f} / {fps:.1f}",
                f"Infer: {t_infer:.1f}ms",
                f"Pre: {t_pre:.1f}ms",
                f"Objects: {count}"
            ]
            for i, text in enumerate(texts):
                color = (0, 255, 0) if i == 0 else (0, 255, 255)
                cv2.putText(vis_frame, text, (30, y_offset + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 显示与保存
            out.write(vis_frame)
            cv2.imshow('YOLOv3-MCU', vis_frame)



            # 打印进度（保留行内刷新，不适合 logger）
            label_info = ", ".join(
                f"{self.class_names[i]}:{(labels == i).sum()}"
                for i in np.unique(labels.astype(int))
            ) if count > 0 else "None"

            print(f"\rFrame {frame_id}/{total_frames} | FPS: {fps_real:.1f} | {label_info}", end="", flush=True)

            # print(f"\rFrame {frame_id}/{total_frames} | FPS: {fps_real:.1f} | Drones: {count}", end="", flush=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print()
                logger.info("Stopped by user.")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Video saved to: {save_name}")


if __name__ == '__main__':
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="YOLOv3-MCUNet Detection")

    # 默认路径配置
    default_video = '/home/verser/Videos/2.mp4'
    default_img = '/home/verser/Pictures/zidane.jpg'  #你可以改成你的测试图片路径
    default_weights = './checkpoints/best_yolov3_mcu.pth'

    parser.add_argument('--source', type=str, default=default_img, help='Path to image or video file')
    parser.add_argument('--weights', type=str, default=default_weights, help='Path to .pth model')
    parser.add_argument('--data', type=str, default='/home/verser/Python/GD_Net/data/coco80.yaml', help='数据集配置 yaml，如 data/udacity.yaml')
    parser.add_argument('--output', type=str, default='./det_results/', help='Directory to save results')
    parser.add_argument('--img-size', type=int, default=320, help='Inference image size')

    opt = parser.parse_args()

    # ==================== 执行 ====================
    detector = MCUNetDetector(
        weights_path=opt.weights,
        data_cfg=opt.data,
        img_size=opt.img_size
    )

    detector.run(opt.source, opt.output)