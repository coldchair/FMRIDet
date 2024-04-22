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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, feature_dim, num_layers, num_heads, hidden_dim, output_dim, dropout=0.1, max_len=7800):
        super().__init__()

        self.pos_encoder = PositionalEncoding(feature_dim, dropout, max_len)
        encoder_layers = TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(feature_dim, output_dim)

    def forward(self, src):
        # src shape: (seq_len, batch_size, feature_dim)
        # print(src)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        encoded_features = self.transformer_encoder(src)
        pooled_output = encoded_features.mean(dim=0)
        output = self.linear(pooled_output)
        return output
