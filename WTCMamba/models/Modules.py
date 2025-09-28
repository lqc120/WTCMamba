import torch
from torch import nn
import math
from models.vmamba import SS2D_brevity
from models.Norm import LayerNorm
from pytorch_wavelets import DWTForward



class SE_Block(nn.Module):
    def __init__(self, channels):
        super(SE_Block, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x1, x2):
        x = self.skip_scale * x1 + self.conv(self.up(x2))
        return x

class DI_Block(nn.Module):
    def __init__(self, channels):
        super(DI_Block, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.down = nn.MaxPool2d(2, stride=2)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x1, x2):
        x = self.conv(self.down(x1)) + self.skip_scale * x2
        return x

class SC_Block(nn.Module):
    def __init__(self, channels):
        super(SC_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.SiLU()
        )
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x1, x1_0):
        x = self.skip_scale * x1 + self.conv(x1_0)
        return x

class SA(nn.Module):
    def __init__(self,):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(cat)
        return self.sigmoid(out)



class CSFP(nn.Module):
    def __init__(self, dec_channels, channels):
        super(CSFP, self).__init__()
        self.fconv_0 = nn.Sequential(
            nn.Conv2d(dec_channels[0], channels, kernel_size=1, stride=1),
            LayerNorm(channels),
        )
        self.fconv_1 = nn.Sequential(
            nn.Conv2d(dec_channels[1], channels, kernel_size=1, stride=1),
            LayerNorm(channels),
        )
        self.fconv_2 = nn.Sequential(
            nn.Conv2d(dec_channels[2], channels, kernel_size=1, stride=1),
            LayerNorm(channels),
        )
        self.fconv_3 = nn.Sequential(
            nn.Conv2d(dec_channels[3], channels, kernel_size=1, stride=1),
            LayerNorm(channels),
        )
        self.se0 = SE_Block(channels)
        self.se1 = SE_Block(channels)
        self.se2 = SE_Block(channels)

        self.sa = SA()
        self.down = nn.MaxPool2d(2, stride=2)
        self.down4 = nn.MaxPool2d(4, stride=4)
        self.down8 = nn.MaxPool2d(8, stride=8)

        self.di0 = DI_Block(channels)
        self.di1 = DI_Block(channels)
        self.di2 = DI_Block(channels)

        self.sc0 = SC_Block(channels)
        self.sc1 = SC_Block(channels)
        self.sc2 = SC_Block(channels)
        self.sc3 = SC_Block(channels)


    def forward(self, x0, x1, x2, x3):
        x3 = self.fconv_3(x3)
        x2 = self.fconv_2(x2)
        x1 = self.fconv_1(x1)
        x0 = self.fconv_0(x0)
        x0_0 = x0
        x1_0 = x1
        x2_0 = x2
        x3_0 = x3
        x2 = self.se2(x2, x3)
        x1 = self.se1(x1, x2)
        x0 = self.se0(x0, x1)
        s0 = self.sa(x0)
        x0 = x0 * s0
        x1 = x1 * self.down(s0)
        x2 = x2 * self.down4(s0)
        x3 = x3 * self.down8(s0)
        x1 = self.di0(x0, x1)
        x2 = self.di1(x1, x2)
        x3 = self.di2(x2, x3)
        x0 = self.sc0(x0, x0_0)
        x1 = self.sc1(x1, x1_0)
        x2 = self.sc2(x2, x2_0)
        x3 = self.sc3(x3, x3_0)
        return x0, x1, x2, x3


class WT_Block(nn.Module):
    def __init__(self, channels):
        super(WT_Block, self).__init__()
        dek = 4
        self.de_conv = nn.Sequential(
            nn.Conv2d(channels, channels // dek, kernel_size=1, stride=1),
            LayerNorm(channels // dek),
        )
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv = nn.ConvTranspose2d(channels//dek * 4, channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.de_conv(x)
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)
        return x

class CVSS_Block(nn.Module):
    def __init__(self, hidden_dim, d_state=16):
        super(CVSS_Block, self).__init__()
        self.d_inner = hidden_dim * 2
        self.in_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, bias=False)
        self.ln_1 = LayerNorm(hidden_dim)
        self.in_proj = nn.Conv2d(hidden_dim, self.d_inner, kernel_size=1, stride=1, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, stride=1, groups=self.d_inner)
        self.act = nn.SiLU()
        self.ss2d = SS2D_brevity(d_model=hidden_dim, d_state=d_state)
        self.out_norm = LayerNorm(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, hidden_dim, kernel_size=1, stride=1, bias=False)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):
        x = self.in_conv(x)
        x = self.ln_1(x)
        input = x
        x = self.in_proj(x)
        x = self.act(self.conv2d(x))
        y = self.ss2d(x)
        y = self.out_norm(y)
        out = self.out_proj(y)
        out = out + input * self.skip_scale
        return out


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return v * x


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)



class PF_Block(nn.Module):
    def __init__(self, channels):
        super(PF_Block, self).__init__()
        self.inter_c = channels//4
        self.norm_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            LayerNorm(channels),
        )
        self.detail_conv1 = nn.Conv2d(self.inter_c, self.inter_c, kernel_size=3, stride=1, padding=1,
                                      groups=self.inter_c, bias=False)
        self.detail_conv2 = nn.Conv2d(self.inter_c, self.inter_c, kernel_size=7, stride=1, padding=3,
                                      groups=self.inter_c, bias=False)
        self.detail_conv3 = nn.Conv2d(self.inter_c, self.inter_c, kernel_size=3, stride=1, padding=2, dilation=2,
                                      groups=self.inter_c, bias=False)
        self.detail_conv4 = nn.Conv2d(self.inter_c, self.inter_c, kernel_size=3, stride=1, padding=3, dilation=3,
                                      groups=self.inter_c, bias=False)
        self.fusion_conv = nn.Sequential(
            LayerNorm(channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.SiLU(),
        )
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        res = self.norm_conv(x)
        x1, x2, x3, x4 = res.chunk(4, dim=1)
        x1 = self.detail_conv1(x1)
        x2 = self.detail_conv2(x1+x2)
        x3 = self.detail_conv3(x2+x3)
        x4 = self.detail_conv4(x3+x4)
        y = self.fusion_conv(torch.cat((x1, x2, x3, x4), dim=1)) + self.skip_scale * res
        return y

class WGM_Module(nn.Module):
    def __init__(self, channels):
        super(WGM_Module, self).__init__()
        self.hw = WT_Block(channels)
        self.mamba = CVSS_Block(channels)
        self.eca = ECA(channels)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.pf = PF_Block(channels)

    def forward(self, x):
        q = self.hw(x)
        q = self.mamba(q)
        x = self.eca(q) + self.skip_scale * x
        x = self.pf(x)
        return x



class DualConv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualConv_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            LayerNorm(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, groups=in_channels),
            LayerNorm(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.SiLU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.conv3(self.conv1(x)+self.conv2(x))
        return x


class CNN_Encoder(nn.Module):
    def __init__(self, channels):
        super(CNN_Encoder, self).__init__()
        self.wgm0 = nn.Sequential(
            DualConv_Block(channels[0], channels[0]),
            DualConv_Block(channels[0], channels[0]),
        )
        self.wgm1 = nn.Sequential(
            DualConv_Block(channels[1], channels[1]),
            DualConv_Block(channels[1], channels[1]),
        )
        self.wgm2 = nn.Sequential(
            DualConv_Block(channels[2], channels[2]),
            DualConv_Block(channels[2], channels[2]),
            DualConv_Block(channels[2], channels[2]),
            DualConv_Block(channels[2], channels[2]),
        )
        self.wgm3 = nn.Sequential(
            DualConv_Block(channels[3], channels[3]),
            DualConv_Block(channels[3], channels[3]),
        )
        self.downsample0 = nn.Conv2d(3, channels[0], kernel_size=4, stride=4)
        self.downsample1 = nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2)
        self.downsample2 = nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2)
        self.downsample3 = nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2)

    def forward(self, x):
        ex0_ = self.wgm0(self.downsample0(x))
        ex1_ = self.wgm1(self.downsample1(ex0_))
        ex2_ = self.wgm2(self.downsample2(ex1_))
        ex3_ = self.wgm3(self.downsample3(ex2_))
        return ex0_, ex1_, ex2_, ex3_



class Mamba_Decoder(nn.Module):
    def __init__(self, channels):
        super(Mamba_Decoder, self).__init__()
        self.wgm0 = WGM_Module(channels)
        self.wgm1 = WGM_Module(channels)
        self.wgm2 = WGM_Module(channels)
        self.wgm3 = WGM_Module(channels)
        self.upsample0 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.skip_scale2 = nn.Parameter(torch.ones(1))
        self.skip_scale1 = nn.Parameter(torch.ones(1))
        self.skip_scale0 = nn.Parameter(torch.ones(1))

    def forward(self, ex0, ex1, ex2, ex3):
        ex3_ = self.upsample3(self.wgm3(ex3))
        ex2_ = self.upsample2(self.wgm2(ex3_ + self.skip_scale2 * ex2))
        ex1_ = self.upsample1(self.wgm1(ex2_ + self.skip_scale1 * ex1))
        ex0_ = self.upsample0(self.wgm0(ex1_ + self.skip_scale0 * ex0))
        return ex0_, ex1_, ex2_, ex3_




