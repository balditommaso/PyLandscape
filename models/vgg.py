"""
Modified from: https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/imagenet_classification/models/vgg.py
"""

from torch import nn, tensor
from typing import *

class VGG(nn.Module):

    def __init__(self, cfg: List, batch_norm: bool, num_classes: int = 1000):
        super(VGG, self).__init__()
        self.features = make_layers(cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=False),
        )
        
        self._initialize_weights()

    def forward(self, x: tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg: List, batch_norm: bool = False):
    layers = []
    in_channels = 3
    for v in cfg:
        
        if v == 'M':
            # add maxpool
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # add convolution and activation
            conv2d = nn.Conv2d(
                in_channels,
                v,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=not batch_norm,
            )
            act = nn.ReLU()
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
            
    return nn.Sequential(*layers)



def VGG_11(num_classes: int):
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    net = VGG(cfg, False, num_classes)
    return net


def VGG_16(num_classes: int):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    net = VGG(cfg, False, num_classes)
    return net