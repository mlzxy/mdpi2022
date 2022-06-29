"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True)
        )

        self.post_residual = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.post_residual(self.residual(x))
        if self.stride == 1 and self.in_channels == self.out_channels:
            return residual + x
        else:
            return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100, size=1.0, use_cifar10=False):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, int(size*32), 1, padding=1),
            nn.BatchNorm2d(int(size*32)),
            nn.ReLU6(inplace=True)
        )
        
        if use_cifar10:
            stride_list = [1, 1, 2, 2, 1, 2, 1]
        else:
            stride_list = [1, 2, 2, 2, 1, 1, 1]

        self.stage1 = LinearBottleNeck(int(size*32), int(size*16), stride_list[0], 1)
        self.stage2 = self._make_stage(2, int(size*16), int(size*24), stride_list[1], 6)
        self.stage3 = self._make_stage(3, int(size*24), int(size*32), stride_list[2], 6)
        self.stage4 = self._make_stage(4, int(size*32), int(size*64), stride_list[3], 6)
        self.stage5 = self._make_stage(3, int(size*64), int(size*96), stride_list[4], 6)
        self.stage6 = self._make_stage(3, int(size*96), int(size*160), stride_list[5], 6)
        self.stage7 = LinearBottleNeck(int(size*160), int(size*320), stride_list[6], 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(int(size*320), int(size*1280), 1),
            nn.BatchNorm2d(int(size*1280)),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(int(size*1280), class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

def mobilenetv2(class_num=100, use_cifar10=False):
    return MobileNetV2(class_num=class_num, use_cifar10=use_cifar10)

def mobilenetv2half():
    return MobileNetV2(size=0.5)

def mobilenetv2quarter():
    return MobileNetV2(size=0.25)