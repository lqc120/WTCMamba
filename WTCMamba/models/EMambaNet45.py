import torch
from torch import nn
import torch.nn.functional as F
from models.Modules import CNN_Encoder, CSFP, Mamba_Decoder, DualConv_Block

class EMambaNet(nn.Module):
    def __init__(self, ):
        super(EMambaNet, self).__init__()
        channels = [48, 96, 192, 384]
        self.backbone = CNN_Encoder(channels)
        self.csfp = CSFP(channels, channels[0])
        self.mamba_decoder = Mamba_Decoder(channels[0])
        self.out3 = nn.Conv2d(channels[0], 1, kernel_size=1, stride=1)
        self.out2 = nn.Conv2d(channels[0], 1, kernel_size=1, stride=1)
        self.out1 = nn.Conv2d(channels[0], 1, kernel_size=1, stride=1)
        self.out0 = nn.Conv2d(channels[0], 1, kernel_size=1, stride=1)
        self.dualConv_block = DualConv_Block(channels[0], 1)

    def forward(self, input):
        size = input.size()
        x0, x1, x2, x3 = self.backbone(input)
        x0_, x1_, x2_, x3_ = self.csfp(x0, x1, x2, x3)
        ex0_, ex1_, ex2_, ex3_ = self.mamba_decoder(x0_, x1_, x2_, x3_)
        f3 = self.out3(ex3_)
        f2 = self.out2(ex2_)
        f1 = self.out1(ex1_)
        f0 = self.out0(ex0_)
        p0 = self.dualConv_block(ex0_)
        p0 = F.interpolate(p0, size=(size[2], size[3]), mode='bilinear')
        return p0, f0, f1, f2, f3



if __name__ == '__main__':
    x = torch.randn(2, 3, 320, 320).cuda()  # B C W H
    slicing = EMambaNet().cuda()
    yout = slicing(x)
    for y in yout:
        print('y', y.shape)

