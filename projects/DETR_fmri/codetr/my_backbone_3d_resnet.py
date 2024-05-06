# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.models.layers import (DetrTransformerDecoder, DetrTransformerEncoder,
                      SinePositionalEncoding)
from mmdet.models.detectors.base_detr import DetectionTransformer
from mmdet.models.detectors.detr import DETR

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

from mmengine.model import BaseModule, ModuleList

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels, stride)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# inputs have shape (bs, 1, n, m, h)
class ResNet3D(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet3D, self).__init__()
        self.in_channels = 256
        self.conv1 = nn.Conv3d(1, self.in_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 1024, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 2048, layers[3], stride=2)

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 零初始化残差连接
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

class ChannelMapper3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelMapper3D, self).__init__()
        # 使用1x1x1的卷积核改变通道数，这样不会改变深度、高度和宽度的大小
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=1, padding=0)
    
    def forward(self, x):
        return self.conv(x)

@MODELS.register_module()
class Backbone_3d(BaseModule):
    def __init__(self, out_channels = 64, **kwargs):
        super().__init__(kwargs)
        self.resnet = ResNet3D(BasicBlock3D, [2, 2, 2, 2])
        self.chm = ChannelMapper3D(in_channels = 512, out_channels=out_channels)
        self.out_channels = out_channels
    
    def forward(self, x): # x shape : [bs, n, m, h]
        # print(x.shape)
        # print(x.device)
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = self.chm(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        return  (x,)