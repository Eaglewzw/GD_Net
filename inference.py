import torch
import cv2
import numpy as np
import time
import os
from gd_net.yolov3_mcu_net import YOLOv3_McuNet
from yolov3_mcu_config import cfg
from utils.vis_tools import visualize
from argparse import Namespace


# ==================== 正确的 letterbox（兼容 PyTorch 2.4+） ====================
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw // 2, dh // 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1  ))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # BGR → RGB, HWC → CHW, /255, add batch
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor, r, (dw, dh)


def detect_video_optimized():
    # video_path = '/media/verser/Verse/02_MyProject/01_UAV_Project/UAV_Vv_Org/01_DJI_UAV_ORG/DJI_0063.mp4'
    video_path = '/home/verser/Pictures/drone.jpg'
    # video_path = '/dev/video0'  # 使用摄像头
    save_path  = './det_results/'
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 256

    # ==================== 模型加载 ====================
    print("正在加载模型权重...")
    ckpt = torch.load('/home/verser/Videos/checkpoints/best_yolov3_mcu.pth', map_location='cpu')  # 先全读到 CPU

    model = YOLOv3_McuNet(
        cfg, 
        device=device, 
        num_classes=1, 
        trainable=False, 
        deploy=False,

    )

    model.load_state_dict(ckpt)        # 先加载权重
    model = model.to(device)           # 再整体搬到 GPU（所有参数自动转为 cuda）
    model.eval()

    print(f"模型加载完成！运行设备: {device}")


    # ==================== 视频 ====================
    cap = cv2.VideoCapture(video_path)
    print(f"视频分辨率: {int(cap.get(3))}×{int(cap.get(4))} @ {cap.get(5):.1f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_name = os.path.join(save_path, 'phantom12_det_final.avi')
    out = cv2.VideoWriter(save_name, fourcc, 30.0, (1920, 1080))

    class_colors = [(0, 255, 255)]  # 亮黄
    class_names = ["drone"]

    frame_id = 0
    fps_counter = FPSCounter(window_size=60)  # 平滑 FPS
    infer_times = []  # 精确推理时间列表
    preprocess_times = []  # 预处理时间
    total_frame_times = []  # 总帧处理时间

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # ==================== 精准计时：预处理 ====================
        t_pre_start = time.time()
        x, ratio, (dw, dh) = letterbox(frame, new_shape=img_size)
        x = x.to(device)
        t_pre_end = time.time()
        preprocess_ms = (t_pre_end - t_pre_start) * 1000
        preprocess_times.append(preprocess_ms)

        # ==================== 精准计时：推理 ====================
        t_infer_start = time.time()
        with torch.no_grad():
            outputs = model(x)
        t_infer_end = time.time()
        infer_ms = (t_infer_end - t_infer_start) * 1000
        infer_times.append(infer_ms)

        # ==================== 解析结果（保持不变） ====================
        bboxes = outputs['bboxes']
        scores = outputs['scores']
        labels = outputs['labels']

        if len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / ratio
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / ratio

        drone_count = len(bboxes)

        # ==================== 精准计时：总帧处理时间 ====================
        t_total_start = time.time()  # 这一帧开始时间（用于 FPS）
        # ... 可视化代码 ...
        t_total_end = time.time()
        total_frame_ms = (t_total_end - t_total_start) * 1000
        total_frame_times.append(total_frame_ms)

        # ==================== 精准 FPS 计算 ====================
        # 1. 滑动窗口平滑 FPS（基于总帧处理时间）
        fps_smooth = fps_counter.tick()

        # 2. 推理专属 FPS（只算推理时间）
        if len(infer_times) > 1:
            recent_infer_avg = np.mean(infer_times[-10:])  # 最近 10 帧平均
            infer_fps = 1000.0 / recent_infer_avg if recent_infer_avg > 0 else 0
        else:
            infer_fps = 0

        # 3. 预处理平均时间
        if len(preprocess_times) > 1:
            preprocess_avg = np.mean(preprocess_times[-10:])
        else:
            preprocess_avg = 0

        # ==================== 可视化（更精准显示） ====================
        frame_vis = visualize(frame.copy(), bboxes, scores, labels.astype(int), 
                            class_colors, class_names, [0])
        frame_show = cv2.resize(frame_vis, (1920, 1080))

        # 精准信息显示
        y_offset = 40
        cv2.putText(frame_show, f"Real FPS: {fps_smooth:.1f}", (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame_show, f"Infer: {infer_ms:.1f}ms (Avg: {np.mean(infer_times[-10:]):.1f}ms)", 
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame_show, f"Preprocess: {preprocess_ms:.1f}ms", 
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        y_offset += 25
        cv2.putText(frame_show, f"Drones: {drone_count}", 
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        out.write(frame_show)
        cv2.imshow('YOLOv3-MCU Precision Detection', frame_show)

        # 精准日志
        print(f"Frame {frame_id:4d} | FPS: {fps_smooth:5.1f} | "
            f"Infer: {infer_ms:5.1f}ms | Pre: {preprocess_ms:4.1f}ms | "
            f"Drones: {drone_count:2d}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n检测完成！视频已保存: {save_name}")

# ==================== FPSCounter 类（滑动窗口） ====================
class FPSCounter:
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

if __name__ == '__main__':
    detect_video_optimized()