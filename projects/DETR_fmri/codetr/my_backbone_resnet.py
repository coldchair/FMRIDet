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

class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual

# bs * channel * 26688 -> bs * 2048 * 834
# while batch_size = 1, batchnorm may cause error
@MODELS.register_module()
class ResNet50_1D(BaseModule):
    def __init__(self, in_channels=1, out_channels = 2048, **kwargs):
        super().__init__(kwargs)
        self.in_channels = in_channels
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            #
            Bottlrneck(256,128,512, True),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            #
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            #
            Bottlrneck(1024,512,out_channels, True),
            Bottlrneck(out_channels,512,out_channels, False),
            Bottlrneck(out_channels,512,out_channels, False),
        )

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-1])
        # print(x.shape)
        x = self.features(x)
        return (x,)
        

@MODELS.register_module()
class Backbone_fmri_resnet1d_2_imgfeat(BaseModule):
    def __init__(self, input_dim = 26688, input_dim_t = 834, output_dim = (256, 25, 25), resnet_blocks = 4, out_channels = 64, hidden_dim = 2048, **kwargs):
        super().__init__(kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.resnet_blocks = resnet_blocks
        self.input_dim_t = input_dim_t

        self.resnet = ResNet50_1D(in_channels=1, out_channels = self.out_channels)
        self.fc = nn.Linear(self.out_channels * self.input_dim_t, self.hidden_dim)
        self.block = nn.ModuleList([
            nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
            ) for _ in range(self.resnet_blocks)]
        )
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim[0]*self.output_dim[1]*self.output_dim[2])

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-1])
        x = self.resnet(x)[0]
        x = x.view(-1, self.out_channels * x.shape[-1])
        x = self.fc(x)
        residual = x
        for block in self.block:
            x = block(x) + residual
            residual = x
        x = self.fc2(x)
        x = x.view(-1, self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return (x,)