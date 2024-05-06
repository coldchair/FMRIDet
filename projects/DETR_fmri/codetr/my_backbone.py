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

from .my_base_modules import TransformerPredictor
from .my_backbone_resnet import Backbone_fmri_resnet1d_2_imgfeat
from .my_backbone_3d_resnet import Backbone_3d
from .my_backbone_vit3d import Backbone_vit3d

# inputs : batch_size * fmri_len
# outputs : channels * 25 * 25
@MODELS.register_module()
class Backbone_fmri_transformer(BaseModule):
    def __init__(self, output_dim = (32, 25, 25), input_dim = 26688,
                 feature_dim = 256, num_layers = 6, num_heads = 8,
                 hidden_dim = 2048, middle_dim = 4096, **kwargs):
                 # middle_dim : The dimension of Transformer output
        super().__init__(kwargs)

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.seq_len = input_dim // feature_dim + (input_dim % feature_dim != 0)
        

        self.transformer = TransformerPredictor(
            self.seq_len, feature_dim, num_layers, num_heads,
            hidden_dim, middle_dim
        )
        self.fc = nn.Linear(middle_dim, output_dim[0]*output_dim[1]*output_dim[2])

    def forward(self, x):
        # padding the input
        if (self.input_dim % self.feature_dim) != 0:
            x = F.pad(x, (0, self.feature_dim - self.input_dim % self.feature_dim,))
        x = x.view(-1, self.seq_len, self.feature_dim)

        x = self.transformer(x)
        x = self.fc(x)
        x = x.view(-1, self.output_dim[0], self.output_dim[1], self.output_dim[2])

        return (x,)

@MODELS.register_module()
class Backbone_fmri(BaseModule):
    def __init__(self, input_dim = 26688, output_dim = (256, 25, 25), resnet_blocks = 4, hidden_dim = 2048, **kwargs):
        super().__init__(kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resnet_blocks = resnet_blocks
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
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
        x = self.fc(x)
        residual = x
        for block in self.block:
            x = block(x) + residual
            residual = x
        x = self.fc2(x)
        x = x.view(-1, self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return (x,)

@MODELS.register_module()
class Backbone_None(BaseModule):
    def __init__(self, bool_2D = True, **kwargs):
        super().__init__(kwargs)
        self.bool_2D = bool_2D
    
    def forward(self, x): # x shape : [bc, length, 64]
        if (self.bool_2D == False):
            x = x.permute(0, 2, 1)
            return (x,)
        else:
            n = x.shape[-2]
            e = x.shape[-1]
            k = 1
            while (k * k < n):
                k += 1
            x = F.pad(x, (0, 0, 0, k * k - n, 0, 0))
            x = x.view(-1, k, k, e)
            x = x.permute(0, 3, 1, 2)
        return (x,)

@MODELS.register_module()
class Backbone_Conv(BaseModule):
    def __init__(self, out_channels = 256, **kwargs):
        super().__init__(kwargs)
        self.out_channels = out_channels
        self.conv = nn.Conv1d(1, out_channels, kernel_size = 32, stride = 16, padding = 0)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return (x,)

@MODELS.register_module()
class Backbone_fmri_seperate(BaseModule):
    def __init__(self, input_dim = 26688, output_dim = (256, 25, 25), resnet_blocks = 4, hidden_dim = 256, **kwargs):
        super().__init__(kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

        for i in range(5):
            for j in range(5):
                setattr(self,
                        "backbone_{}_{}".format(i, j),
                        Backbone_fmri(input_dim = input_dim,
                                      output_dim = (output_dim[0], 5, 5),
                                      resnet_blocks = resnet_blocks,
                                      hidden_dim = hidden_dim))

    def forward(self, x):
        outputs = []
        for i in range(5):
            for j in range(5):
                outputs.append(getattr(self, "backbone_{}_{}".format(i, j))(x)[0].unsqueeze(1))
        outputs = torch.cat(outputs, dim = 1).view(-1, 5, 5, self.output_dim[0], 5, 5)
        outputs = outputs.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, self.output_dim[0], 25, 25)
        return (outputs,)
        
                