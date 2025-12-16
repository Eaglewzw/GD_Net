import torch
import cv2
import numpy as np
import time
import os
import argparse
from pathlib import Path

# 假设这些是你项目中的模块
from gd_net.yolov3_mcu_net import YOLOv3_McuNet
from yolov3_mcu_config import cfg
from utils.vis_tools import visualize


# ==================== 工具函数 & 类 ====================

def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """ 保持长宽比缩放图片，并进行 padding """
    h, w = img.shape[:2]
    # 如果 new_shape 是整数，则转换为 (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # BGR → RGB, HWC → CHW, /255, add batch
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor, r, (dw, dh)


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
    def __init__(self, weights_path, img_size=256, device=None):
        self.img_size = img_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        print(f"🚀 Loading model from {weights_path} to {self.device}...")
        self.model = YOLOv3_McuNet(cfg, device=self.device, num_classes=1, trainable=False)

        ckpt = torch.load(weights_path, map_location='cpu')
        # 兼容 state_dict 可能包含 'model' 键的情况
        if 'model' in ckpt:
            ckpt = ckpt['model']

        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 预热模型 (Warmup) - 避免第一次推理时间过长
        print("🔥 Warming up...")
        dummy_input = torch.zeros(1, 3, img_size, img_size).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)
        print("✅ Model loaded and ready.")

        self.class_colors = [(0, 255, 255)]  # 亮黄
        self.class_names = ["drone"]

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
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / ratio
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / ratio

        # 4. 可视化绘制
        vis_frame = visualize(frame.copy(), bboxes, scores, labels.astype(int),
                              self.class_colors, self.class_names, [0])

        return vis_frame, t_pre, t_infer, len(bboxes)

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
            print(f"❌ Unsupported file format: {file_ext}")

    def _process_image(self, img_path, save_dir):
        print(f"🖼️ Processing Image: {img_path}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("❌ Failed to load image.")
            return

        # 执行推理
        vis_frame, t_pre, t_infer, count = self.infer_single_frame(frame)

        # 绘制文本信息 (不需要FPS)
        info_text = f"Drones: {count} | Infer: {t_infer:.1f}ms"
        cv2.putText(vis_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # 保存
        save_name = os.path.join(save_dir, img_path.name)
        cv2.imwrite(save_name, vis_frame)
        print(f"✅ Saved to: {save_name} | {info_text}")

    def _process_video(self, vid_path, save_dir):
        print(f"🎥 Processing Video: {vid_path}")
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
            vis_frame, t_pre, t_infer, count = self.infer_single_frame(frame)

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

            # 打印进度
            print(f"\rFrame {frame_id}/{total_frames} | FPS: {fps_real:.1f} | Drones: {count}", end="")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Stopped by user.")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Video saved to: {save_name}")


if __name__ == '__main__':
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="YOLOv3-MCUNet Detection")

    # 默认路径配置
    default_video = '/media/verser/robot/Dataset/ARD-MAV/videos/phantom28.mp4'
    default_img = '/home/verser/Pictures/DJI.jpg'  # 你可以改成你的测试图片路径
    default_weights = './checkpoints/best_yolov3_mcu.pth'

    parser.add_argument('--source', type=str, default=default_img, help='Path to image or video file')
    parser.add_argument('--weights', type=str, default=default_weights, help='Path to .pth model')
    parser.add_argument('--output', type=str, default='./det_results/', help='Directory to save results')
    parser.add_argument('--img-size', type=int, default=256, help='Inference image size')

    opt = parser.parse_args()

    # ==================== 执行 ====================
    detector = MCUNetDetector(
        weights_path=opt.weights,
        img_size=opt.img_size
    )

    detector.run(opt.source, opt.output)