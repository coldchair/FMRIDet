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
from mmdet.models.detectors.base import BaseDetector
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
import math
from typing import Optional
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptMultiConfig


@MODELS.register_module()
class SinePositionalEncoding1D(BaseModule):
    """Position encoding with sine and cosine functions for 1D data."""

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor, input: Optional[Tensor] = None) -> Tensor:
        """Forward function for `SinePositionalEncoding1D`."""
        assert not (mask is None and input is None)

        if mask is not None:
            B, L = mask.size()
            device = mask.device
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # logical_not
            x_embed = not_mask.cumsum(1, dtype=torch.float32)
        else:
            # single sequence or batch sequence with no padding
            B, _, L = input.shape
            device = input.device
            x_embed = torch.arange(1, L + 1, dtype=torch.float32, device=device).view(1, -1).repeat(B, 1)

        if self.normalize:
            x_embdd = (x_embed + self.offset) / (x_embed[:, :] + self.eps) * self.scale

        dim_t = torch.arange(
            self.num_feats * 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_feats * 2))

        pos_x = x_embdd[:, :, None] / dim_t

        pos_x[:, :, 0::2] = pos_x[:, :, 0::2].sin()
        pos_x[:, :, 1::2] = pos_x[:, :, 1::2].cos()
        pos = pos_x.permute(0, 2, 1)

        # print(pos.shape)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str