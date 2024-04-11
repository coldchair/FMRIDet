# Copyright (c) OpenMMLab. All rights reserved.
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType

from mmdet.models.data_preprocessors import DetDataPreprocessor

try:
    import skimage
except ImportError:
    skimage = None

import torch


@MODELS.register_module()
class DetDataPreprocessor_fmri(DetDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:
        results = super().forward(data, training)
        results['inputs_fmri'] = torch.stack(data['inputs_fmri'])
        return results
    