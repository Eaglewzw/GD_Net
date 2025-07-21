import torch
import torch.nn as nn

from gd_net_basic import Conv

class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=5,act_type='lrelu',norm='BN'):
        super().__init__()
        inter_dim = int(expand_ratio * in_dim)
        self.out_dim = out_dim
        self.cv1=Conv(in_dim, inter_dim, k=1, act_type=act_type, norm=norm_type)
        self.cv2=Conv(inter_dim,4,out_dim,k=1,act_type=act_type,norm_type=norm_type)
        self.m=nn.MaxPool2d(kernel_size=pooling_size, stride=1,padding=pooling_size//2)
    def forward(self, x):
        x=self.cv1(x)
        y1=self.m(x)
        y2=self.m(y1)
        return self.cv2(torch.cat((x,y1,y2,self.m(y2)),1))


