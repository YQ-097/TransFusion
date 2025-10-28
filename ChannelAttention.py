import numpy as np
import torch
from torch import nn
from torch.nn import init



class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)# nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output




class ChannelAttentionBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca_ir = ChannelAttention(channel=channel,reduction=reduction)
        self.ca_vis = ChannelAttention(channel=channel, reduction=reduction)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, I_ir , I_vis):
        W_ir = self.ca_ir(I_ir)
        W_vis = self.ca_vis(I_vis)
        out_ir=I_ir + I_ir*self.sigmoid(2 * W_ir -  W_vis)
        out_vis = I_vis + I_vis * self.sigmoid(2 * W_vis - W_ir)
        return out_ir, out_vis


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    kernel_size=input.shape[2]
    cbam = ChannelAttentionBlock(channel=512,reduction=16,kernel_size=kernel_size)
    output=cbam(input)
    print(output.shape)

    