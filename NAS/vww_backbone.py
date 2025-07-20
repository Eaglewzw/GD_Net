import torch
import torch.nn as nn
from mcunet.model_zoo import build_model


import time
from thop import profile


# 输入图像预处理
from PIL import Image
from torchvision import transforms


class MCUNetBackbone(nn.Module):
    def __init__(self, checkpoint_path=None, net_id="mcunet-10fps-vww"):
        super().__init__()
        # 原始模型
        self.model, self.image_size, _ = build_model(net_id=net_id, pretrained=False)

        # 提取主干结构（首卷积 + blocks）
        self.first_conv = self.model.first_conv
        self.blocks = self.model.blocks

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        return x  # 返回最后一层 block 的输出

    def get_image_size(self):
        return self.image_size

if __name__ == "__main__":
    # 创建模型并加载权重
    # backbone = MCUNetBackbone("/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth")
    # backbone = MCUNetBackbone("/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth")
    backbone = MCUNetBackbone("/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth")



    transform = transforms.Compose([
        transforms.Resize(backbone.get_image_size()),
        transforms.CenterCrop(backbone.get_image_size()),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = transform(Image.open("/home/verse/Pictures/mine.jpg").convert("RGB")).unsqueeze(0)
    dummy_input = torch.randn(1, 3, backbone.image_size, backbone.image_size)

    with torch.no_grad():
        features = backbone(img)
        print("Feature shape:", features.shape)

        t0 = time.time()
        outputs = backbone(dummy_input)
        t1 = time.time()
        print("Inference Time: {:.4f} seconds".format(t1 - t0))
        print("Output shape:", outputs.shape)

        flops, params = profile(backbone, inputs=(dummy_input,), verbose=False)
        print('==============================')
        print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
        print('Params : {:.2f} M'.format(params / 1e6))