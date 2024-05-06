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

from .my_base_modules import VisionTransformer_3D

@MODELS.register_module()
class Backbone_vit3d(BaseModule):
    def __init__(self,
                 input_size = (42, 46, 61),
                 out_channels = 64,
                 **kwargs):
        super().__init__(kwargs)
        self.input_size = input_size
        self.out_channels = out_channels

        self.vit = VisionTransformer_3D(patch_size=8, width=256, layers=6, heads=8,
                                        input_resolution=input_size,
                                        num_class_embeddings=25)
        for i in range(5):
            for j in range(5):
                setattr(self,
                        "backbone_{}_{}".format(i, j),
                        nn.Linear(256, out_channels * 5 * 5)
                )
    
    def forward(self, x): # x shape : [bs, n, m, h]
        # print(x.shape)
        # print(x.device)
        x = x.unsqueeze(1)
        x = self.vit(x) # x shape : [bs, 25, 256]
        outputs = []
        for i in range(5):
            for j in range(5):
                outputs.append(getattr(self,
                                       "backbone_{}_{}".format(i, j))(x[:, i * 5 + j]).view(
                                           -1, self.out_channels, 5, 5
                                       ))
        outputs = torch.cat(outputs, dim = 1).view(-1, 5, 5, self.out_channels, 5, 5)
        # print(outputs.shape)
        outputs = outputs.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, self.out_channels, 25, 25)
        return  (outputs,)

@MODELS.register_module()
class Backbone_vit3d_2(BaseModule):
    def __init__(self,
                 input_size = (42, 46, 61),
                 out_channels = 64,
                 **kwargs):
        super().__init__(kwargs)
        self.input_size = input_size
        self.out_channels = out_channels

        self.vit = VisionTransformer_3D(patch_size=8, width=512, layers=6, heads=8,
                                        input_resolution=input_size,
                                        num_class_embeddings=10)
        self.Linear = nn.Linear(512 * 10, out_channels * 25 * 25)
    
    def forward(self, x): # x shape : [bs, n, m, h]
        # print(x.shape)
        # print(x.device)
        x = x.unsqueeze(1)
        x = self.vit(x) # x shape : [bs, 25, 256]
        x = x.view(-1, 512 * 10)
        outputs = self.Linear(x).view(-1, self.out_channels, 25, 25)
        return  (outputs,)