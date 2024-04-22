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
from torch import tensor

def get_gaussian_mask(w, bboxes : tensor, n = 25, device = 'cuda'):
    m = bboxes.size(0)
    
    # 防止空的bounding boxes张量
    if m == 0:
        return w
    
    # 计算中心点并归一化
    centers = (bboxes[:, 2:] + bboxes[:, :2]) / 2 / 32
    
    # 创建网格以计算高斯
    xv, yv = torch.meshgrid(torch.linspace(0, n - 1, n), torch.linspace(0, n - 1, n))
    xv = xv.to(device).float()
    yv = yv.to(device).float()
    
    # 扩展网格和中心点张量以便向量化操作
    xv = xv.unsqueeze(0).expand(m, -1, -1)
    yv = yv.unsqueeze(0).expand(m, -1, -1)
    x_centers = centers[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, n, n)
    y_centers = centers[:, 1].unsqueeze(1).unsqueeze(2).expand(-1, n, n)

    
    # 批量计算高斯函数
    gaussians = torch.exp(-((xv - x_centers) ** 2 + (yv - y_centers) ** 2) / 4)

    # 找到每个位置的最大高斯响应
    w.copy_(gaussians.max(dim=0).values) 

    return w

@MODELS.register_module()
class DetDataPreprocessor_fmri(DetDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super(DetDataPreprocessor, self).forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        inputs_fmri = torch.stack(data['inputs_fmri'])

        # print("inputs adsas : ", inputs.device)
        # for x in data['inputs_fmri']:
        #     print(x.device)

        gm = None
        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)
            
            gm = torch.zeros((inputs_fmri.size(0), 25, 25),
                             device = inputs_fmri.device,
                             requires_grad=False)
            for i in range(len(data_samples)):
                # print(data_samples[i])
                bboxes = data_samples[i].gt_instances.bboxes
                get_gaussian_mask(gm[i], bboxes, device = inputs_fmri.device)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, inputs_fmri, gm, data_samples = batch_aug(inputs, inputs_fmri, gm, data_samples)
        
        # print(inputs_fmri.shape)
        # print(inputs_fmri.device)
            
        return {'inputs': inputs,
                'inputs_fmri' : inputs_fmri,
                'gm' : gm,
                'data_samples': data_samples}
    