import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for PyTorch"""
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

class DBFPN(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.attention = attention

        # 1. 输入变换 (Inception-like 1x1 Conv)
        self.in2_conv = nn.Conv2d(in_channels=in_channels[0], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels=in_channels[1], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels=in_channels[2], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels=in_channels[3], out_channels=self.out_channels, kernel_size=1, bias=False)

        # 2. 注意力模块 (可选)
        if self.attention:
            self.p5_attention = SEBlock(self.out_channels)
            self.p4_attention = SEBlock(self.out_channels)
            self.p3_attention = SEBlock(self.out_channels)
            self.p2_attention = SEBlock(self.out_channels)

        # 3. 输出变换 (Final Prediction)
        self.p2_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p5_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        c2, c3, c4, c5 = x

        # 1. 统一通道数
        in2 = self.in2_conv(c2)
        in3 = self.in3_conv(c3)
        in4 = self.in4_conv(c4)
        in5 = self.in5_conv(c5)

        # 2. 自顶向下路径 (Top-down pathway)
        out5 = in5
        out4 = in4 + F.interpolate(out5, scale_factor=2, mode='nearest')
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode='nearest')
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode='nearest')

        # 3. 注意力处理 (如果开启)
        if self.attention:
            out5 = self.p5_attention(out5)
            out4 = self.p4_attention(out4)
            out3 = self.p3_attention(out3)
            out2 = self.p2_attention(out2)

        # 4. 输出卷积处理
        p5 = self.p5_conv(out5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        # 5. 上采样并拼接
        p3 = F.interpolate(p3, scale_factor=2, mode='nearest')
        p4 = F.interpolate(p4, scale_factor=4, mode='nearest')
        p5 = F.interpolate(p5, scale_factor=8, mode='nearest')

        fuse = torch.cat([p2, p3, p4, p5], dim=1)
        
        return fuse