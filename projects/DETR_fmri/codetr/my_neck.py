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
class ChannelMapper_empty(BaseModule):
    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        return inputs
