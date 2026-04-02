import cv2
import numpy as np
import torch


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """保持长宽比缩放图片并进行 padding（letterbox）。

    Args:
        img: BGR numpy array, shape (H, W, 3)
        new_shape: 目标尺寸，int 或 (h, w) tuple
        color: padding 填充颜色 (BGR)

    Returns:
        tensor: float32 tensor, shape (1, 3, H, W), 值域 [0, 1]
        ratio: 缩放比例
        (dw, dh): 左/上方向 padding 像素数
    """
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) // 2
    dh = (new_shape[0] - new_unpad[1]) // 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    # BGR → RGB, HWC → CHW, /255, add batch dim
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor, r, (dw, dh)
