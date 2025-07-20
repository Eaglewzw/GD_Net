
from torch import nn
from tinynas.nn.networks import ProxylessNASNets
import json
import os.path as osp
import torch

def _make_backbone():
    json_file = osp.join(osp.split(osp.abspath(__file__))[0], "/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.json")
    with open(json_file) as f:
        config = json.load(f)
    _model = ProxylessNASNets.build_from_config(config)
    ckpt = torch.load(osp.join(osp.split(osp.abspath(__file__))[0], "/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth"))
    _model.load_state_dict(ckpt['state_dict'], strict=False)
    return _model

class vww_Backbone(nn.Module):
    def x1_hook(self, module, input, output):
        self.x1 = output

    def x2_hook(self, module, input, output):
        self.x2 = output

    def __init__(self, weight_path=None, dummy_scale=1.0):
        super().__init__()
        self.model = _make_backbone()
        self.model.blocks[5].register_forward_hook(self.x1_hook)
        self.model.blocks[10].register_forward_hook(self.x2_hook)


    def forward(self, x):
        x = self.model(x)
        # return [self.x1, self.x2, x]
        return {"dark3": self.x1, "dark4": self.x2, "dark5": x}


if __name__ == "__main__":
    import torch
    input = torch.randn(1, 3, 160, 160)
    # model = CSPDarknet(0.33, 0.25, depthwise=True)
    model = vww_Backbone(0.33, 0.25)
    output = model(input)
    for k in output:
        print(output[k].shape)
    # print(model)


    dtype = next(model.parameters()).dtype

    if dtype == torch.float16:
        print("模型是 FP16 (半精度)")
    elif dtype == torch.float32:
        print("模型是 FP32 (单精度)")
    else:
        print(f"未知数据类型: {dtype}")