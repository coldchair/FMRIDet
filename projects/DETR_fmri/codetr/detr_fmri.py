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

@MODELS.register_module()
class DETR_teacher(DETR):
    def forward(self,
                inputs: torch.Tensor,
                inputs_fmri: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        return super().forward(inputs, data_samples, mode)

@MODELS.register_module()
class Backbone_fmri(BaseModule):
    def __init__(self, input_dim = 26688, output_dim = (256, 25, 25), resnet_blocks = 4, hidden_dim = 2048, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resnet_blocks = resnet_blocks
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15)
            ) for _ in range(self.resnet_blocks)]
        )
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim[0]*self.output_dim[1]*self.output_dim[2])

    def forward(self, x):
        x = self.fc(x)
        residual = x
        for block in self.block:
            x = block(x) + residual
        x = self.fc2(x)
        x = x.view(-1, self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return (x,)
        

@MODELS.register_module()
class DETR_student(DETR):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = DetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def forward(self,
                inputs: torch.Tensor,
                inputs_fmri: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        return super().forward(inputs_fmri, data_samples, mode)