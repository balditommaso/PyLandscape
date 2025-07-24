"""
Modified from: https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/imagenet_classification/models/mobilenetv1.py
"""
from torch import nn, tensor
from typing import *

class DwsConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
        )
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )


    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x



class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bn_eps: float = 1e-5,
    ) -> nn.Module:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = nn.ReLU()


    def forward(self, x: tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x



class MobileNet(nn.Module):

    def __init__(
        self,
        channels: Tuple,
        first_stage_stride: bool,
        first_layer_stride: int = 2,
        in_channels: int = 3,
        num_classes: int = 10
    ) -> nn.Module:
        super(MobileNet, self).__init__()
        init_block_channels = channels[0][0]

        self.features = nn.Sequential()
        init_block = ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=first_layer_stride
        )
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                mod = DwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(in_channels, num_classes, bias=True)


    def forward(self, x: tensor):
        x = self.features(x)
        x = self.final_pool(x)
        x = self.flatten(x)
        out = self.output(x)
        return out



def mobilenet_v1(num_classes: int):
    channels = [
        [32], 
        [64], 
        [128, 128], 
        [256, 256], 
        [512, 512, 512, 512, 512, 512], 
        [1024, 1024]
    ]
    first_stage_stride = False
    first_layer_stride = 1
    net = MobileNet(
        channels=channels, 
        first_stage_stride=first_stage_stride, 
        first_layer_stride=first_layer_stride,
        num_classes=num_classes
    )

    return net