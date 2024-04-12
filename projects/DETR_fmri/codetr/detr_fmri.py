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

@MODELS.register_module()
class DETR_teacher(DETR):
    def forward(self,
                inputs: torch.Tensor,
                inputs_fmri: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        return super().forward(inputs, data_samples, mode)
        

@MODELS.register_module()
class DETR_student(DETR):
    def forward(self,
                inputs: torch.Tensor,
                inputs_fmri: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        return super().forward(inputs_fmri, data_samples, mode)