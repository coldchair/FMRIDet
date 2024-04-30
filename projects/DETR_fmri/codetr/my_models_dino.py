
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
from .my_backbone import Backbone_fmri_transformer, Backbone_fmri
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
from mmdet.models.necks import ChannelMapper
from mmdet.models.detectors import DABDETR
from mmdet.models.layers import  (DABDetrTransformerDecoder,
                                  DABDetrTransformerEncoder, inverse_sigmoid)
from .my_models_position import SinePositionalEncoding1D

from mmdet.models.detectors import DINO


@MODELS.register_module()
class DINO_up(DINO):
    # This function is used to get loss from the model in the validation step
    def val_loss_step(self, data: Union[tuple, dict, list]):
        data = self.data_preprocessor(data, True)
        return self._run_forward(data, mode='loss')  # type: ignore