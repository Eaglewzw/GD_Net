import torch.nn as nn
from gd_net_basic import Conv
from gd_net_basic import ResBlock



class DarkNet53(nn.Module):
    def __init__(self, act_type='silu', norm_type='BN'):
        super(DarkNet53, self).__init__()
        self.feat_dims=[256, 512, 1024]

        #P1
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1, act_type=act_type, norm_type=norm_type),
            Conv(32, 64, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ResBlock(64, 64, nblocks=1, act_type=act_type, norm_type=norm_type),
        )

        #P2
        self.layer_2 = nn.Sequential(
            Conv(64, 128, k=3, p=1, act_type=act_type, norm_type=norm_type),
            ResBlock(128, 128, nblocks=2, act_type=act_type, norm_type=norm_type),
        )

        #p3
        self.layer_3 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type),
            ResBlock(256, 256, nblocks=8, act_type=act_type, norm_type=norm_type),

        )

        #p4
        self.layer_4 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ResBlock(512, 512, nblocks=8, act_type=act_type, norm_type=norm_type),
        )

        #P5
        self.layer_5 = nn.Sequential(
            Conv(512, 1024, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ResBlock(1024, 1024, nblocks=4, act_type=act_type, norm_type=norm_type),
        )

        def forward(self, x):
            c1=self.layer_1(x)
            c2=self.layer_2(c1)
            c3=self.layer_3(c2)
            c4=self.layer_4(c3)
            c5=self.layer_5(c4)

            output=[c3, c4, c5]
            return output

