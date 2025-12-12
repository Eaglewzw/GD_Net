#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

def _make_backbone():
    if __name__ == "__main__":
        from tinynas.nn.networks import ProxylessNASNets
    else:
        from .tinynas.nn.networks import ProxylessNASNets
    import json
    import os.path as osp
    import torch
    json_file = osp.join(osp.split(osp.abspath(__file__))[0], "mcunet-10fps_vww.json")
    with open(json_file) as f:
        config = json.load(f)
    _model = ProxylessNASNets.build_from_config(config)
    ckpt = torch.load(osp.join(osp.split(osp.abspath(__file__))[0], "mcunet-10fps_vww.pth"))
    _model.load_state_dict(ckpt['state_dict'], strict=False)
    return _model


class MyBackbone(nn.Module):
    def x1_hook(self, module, input, output):
        self.x1 = output

    def x2_hook(self, module, input, output):
        self.x2 = output

    def __init__(
            self,
            dep_mul,
            wid_mul,
            out_features=("dark3", "dark4", "dark5"),
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        self.model = _make_backbone()
        # self.model.blocks[8].register_forward_hook(self.x1_hook)
        # self.model.blocks[16].register_forward_hook(self.x2_hook)
        self.model.blocks[10].register_forward_hook(self.x1_hook)
        self.model.blocks[13].register_forward_hook(self.x2_hook)


    def forward(self, x):
        x = self.model(x)
        return {"dark3": self.x1, "dark4": self.x2, "dark5": x}


if __name__ == "__main__":
    import torch
    input = torch.randn(8, 3, 160, 128)
    # model = CSPDarknet(0.33, 0.25, depthwise=True)
    model = MyBackbone(0.33, 0.25, depthwise=True)
    output = model(input)
    for k in output:
        print(output[k].shape)
    # print(model)
