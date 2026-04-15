"""
https://arxiv.org/abs/1611.05431
official code:
https://github.com/facebookresearch/ResNeXt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class BasicBlock_C(nn.Module):
    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, 1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU()),
                ('conv3_0', nn.Conv2d(inner_width, inner_width, 3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU()),
                ('conv1_1', nn.Conv2d(inner_width, inner_width * self.expansion, 1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inner_width * self.expansion))
            ]
        ))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, inner_width * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_width * self.expansion)
            )

    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=5, **kwargs):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion
        
        kernel_size = kwargs.pop('kernel_size', 7)
        stride = kwargs.pop('stride', 2)
        padding = kwargs.pop('padding', 3)
        
        self.conv0 = nn.Conv2d(3, self.in_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=self._make_layer(num_blocks[0],1)
        self.layer2=self._make_layer(num_blocks[1],2)
        self.layer3=self._make_layer(num_blocks[2],2)
        self.layer4=self._make_layer(num_blocks[3],2)
        self.linear = nn.Linear(self.cardinality * self.bottleneck_width, num_classes)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.pool0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.linear(out)
        return out

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

def resnext50_32x4d(num_classes=5, **kwargs):
    num_blocks = kwargs.pop('num_blocks', [3, 4, 6, 3])
    cardinality = kwargs.pop('cardinality', 32)
    bottleneck_width = kwargs.pop('bottleneck_width', 4)
    return ResNeXt(num_blocks=num_blocks, cardinality=cardinality, bottleneck_width=bottleneck_width, num_classes=num_classes, **kwargs)

def resnext101_32x8d(num_classes=5, **kwargs):
    num_blocks = kwargs.pop('num_blocks', [3, 4, 23, 3])
    cardinality = kwargs.pop('cardinality', 32)
    bottleneck_width = kwargs.pop('bottleneck_width', 8)
    return ResNeXt(num_blocks=num_blocks, cardinality=cardinality, bottleneck_width=bottleneck_width, num_classes=num_classes, **kwargs)