# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig

from mmdet.models.necks import ChannelMapper

@MODELS.register_module()
class ChannelMapper_None(BaseModule):
    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        return inputs

@MODELS.register_module()
class ChannelMapper_hiddenlayer(ChannelMapper):
    def __init__(self,
                 hidden_size = 8192,
                 input_size = (32, 25, 25),
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.l1 = nn.Linear(input_size[0] * input_size[1] * input_size[2], hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.l2 = nn.Linear(hidden_size, input_size[0] * input_size[1] * input_size[2])
    
    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        inputs = super().forward(inputs)
        print(inputs[0].shape)
        inputs = inputs[0].view(inputs[0].shape[0], -1)
        inputs = self.l1(inputs)
        inputs = self.ln(inputs)
        inputs = self.l2(inputs)
        inputs = inputs.view(inputs.shape[0], *self.input_size)
        return (inputs,)
