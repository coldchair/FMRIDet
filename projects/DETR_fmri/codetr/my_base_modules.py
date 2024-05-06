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
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 在对数空间计算一次位置编码.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, seq_len, feature_dim, num_layers, num_heads, hidden_dim, output_dim, dropout=0.1, max_len=200):
        super().__init__()

        self.pos_encoder = PositionalEncoding(feature_dim, dropout, max_len)
        encoder_layers = TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(seq_len * feature_dim, output_dim)
        self.ln_post = nn.LayerNorm(output_dim)

    def forward(self, src):
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        encoded_features = self.transformer_encoder(src)
        encoded_features = encoded_features.transpose(0, 1)
        encoded_features = encoded_features.reshape(encoded_features.size(0), -1)
        outputs = self.linear(encoded_features) # shape = [*, output_dim]
        outputs = self.ln_post(outputs) # layer normalization, for better training
        return outputs

# --------------------------------- CLIP ViT --------------------------------- #

from collections import OrderedDict
from typing import Tuple, Union

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class VisionTransformer_3D(nn.Module):
    def __init__(self,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 input_resolution = (42, 46, 61),
                 num_class_embeddings = 25,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_class_embeddings = num_class_embeddings

        padding = patch_size // 2 + 1
        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               padding=padding,
                               bias=False)

        scale = width ** -0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        self.class_embedding = nn.Parameter(scale * torch.randn(num_class_embeddings, width))

        n = []
        for i in range(3):
            n.append((input_resolution[i] + 2 * padding - patch_size) // patch_size + 1)
        
        print(f'input_resolution = {input_resolution}')
        print(f'After conv1, n = {n}')
        
        self.positional_embedding = nn.Parameter(scale * torch.randn(n[0] * n[1] * n[2] + num_class_embeddings,
                                                                     width))

        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],
                                                                      self.num_class_embeddings,
                                                                      x.shape[-1],
                                                                      dtype=x.dtype,
                                                                      device=x.device),
                       x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, :self.num_class_embeddings, :])

        return x
