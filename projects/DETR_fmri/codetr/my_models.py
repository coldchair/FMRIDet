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
from .my_models_dabdetr import DABDETR_distill
from .my_models_position import SinePositionalEncoding1D

# I add some new functions to the DETR model, for some needs in my project
@MODELS.register_module()
class DETR_upgradation(DETR):
    def freeze_params(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    # This function is used to get loss from the model in the validation step
    def val_loss_step(self, data: Union[tuple, dict, list]):
        data = self.data_preprocessor(data, True)
        return self._run_forward(data, mode='loss')  # type: ignore

@MODELS.register_module()
class DETR_teacher(DETR_upgradation):
    def freeze_params(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def init_weights(self) -> None:
        return super(DETR, self).init_weights()

    def forward(self,
                inputs: torch.Tensor,
                inputs_fmri: torch.Tensor,
                gm : torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        return super().forward(inputs, data_samples, mode)


@MODELS.register_module()
class DETR_student(DETR_upgradation):
    def init_weights(self) -> None:
        return super(DETR, self).init_weights()

    def forward(self,
                inputs: torch.Tensor,
                inputs_fmri: torch.Tensor,
                gm : torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        return super().forward(inputs_fmri, data_samples, mode)

import numpy as np
from torch import tensor

@MODELS.register_module()
class DETR_distill(BaseDetector):
    def val_loss_step(self, data: Union[tuple, dict, list]):
        data = self.data_preprocessor(data, True)
        return self._run_forward(data, mode='loss')  # type: ignore

    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 loss_feature_distill_alpha: float = 0.0,
                 loss_encoded_feature_distill_alpha : float = 0.0,
                 loss_feature_type : str = 'L2' or 'gussian mask L2' or 'gussian mask L1',
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher : DETR_teacher = MODELS.build(teacher_cfg)
        self.student : DETR_student = MODELS.build(student_cfg)
        self.teacher.init_weights()
        self.student.init_weights()
        self.teacher.freeze_params()
        self.loss_feature_distill_alpha = loss_feature_distill_alpha
        self.loss_encoded_feature_distill_alpha = loss_encoded_feature_distill_alpha
        self.loss_feature_type = loss_feature_type

    def forward(self,
                inputs: torch.Tensor,
                inputs_fmri: torch.Tensor,
                gm : torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:

            
        if mode == 'loss':
            loss = self.student.loss(inputs_fmri, data_samples)

            # --------------------------------- feat_loss -------------------------------- #
            if (self.loss_feature_distill_alpha > 1e-9):
                if (self.loss_feature_type == 'L2'):
                    feature_teacher = self.teacher.extract_feat(inputs)[0]
                    feature_student = self.student.extract_feat(inputs_fmri)[0]
                    feature_distill_loss = F.mse_loss(feature_teacher, feature_student)
                    loss['feature_distill_loss'] = self.loss_feature_distill_alpha * feature_distill_loss
                elif (self.loss_feature_type == 'gussian mask L2'):
                    feature_teacher = self.teacher.extract_feat(inputs)[0]
                    feature_student = self.student.extract_feat(inputs_fmri)[0]

                    feature_distill_loss = torch.mean((gm.unsqueeze(1) * ((feature_teacher - feature_student) ** 2))) * self.loss_feature_distill_alpha

                    loss['feature_distill_loss'] = feature_distill_loss

            # ----------------------------- encoded_feat_loss ---------------------------- #
            if (self.loss_encoded_feature_distill_alpha > 1e-9):
                encoder_inputs_dict_t, decoder_inputs_dict_t = self.teacher.pre_transformer(
                    (feature_teacher,), data_samples)
                encoder_outputs_dict_t = self.teacher.forward_encoder(**encoder_inputs_dict_t)
                encoder_inputs_dict_s, decoder_inputs_dict_s = self.student.pre_transformer(
                    (feature_student,), data_samples)
                encoder_outputs_dict_s = self.student.forward_encoder(**encoder_inputs_dict_s)
                encoded_feature_distill_loss = F.mse_loss(encoder_outputs_dict_t['memory'], encoder_outputs_dict_s['memory'])
                loss['encoded_feature_distill_loss'] = self.loss_encoded_feature_distill_alpha * encoded_feature_distill_loss

            return loss
        elif mode == 'predict':
            return self.student.predict(inputs_fmri, data_samples)
        elif mode == 'tensor':
            return self.student._forward(inputs_fmri, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    # The following methods are not necessary to implement

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass

@MODELS.register_module()
class DETR_distill_new(DETR_distill):
    def __init__(self,
                feature_distill_size = 64, # = student_transformer_size
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_distill_size = feature_distill_size
        self.feature_channel_mapper = ChannelMapper(in_channels=[self.feature_distill_size],
                                                    out_channels=256,
                                                    kernel_size=1,
                                                    act_cfg=None,
                                                    norm_cfg=None,
                                                    num_outs=1)

    def forward(self,
                inputs: torch.Tensor,
                inputs_fmri: torch.Tensor,
                gm : torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            loss = self.student.loss(inputs_fmri, data_samples)
            # --------------------------------- feat_loss -------------------------------- #
            if (self.loss_feature_distill_alpha > 1e-9):
                if (self.loss_feature_type == 'L2'):
                    feature_teacher = self.teacher.extract_feat(inputs)[0]
                    feature_student = self.student.extract_feat(inputs_fmri)[0]
                    feature_distill_loss = F.mse_loss(feature_teacher, feature_student_mapped)
                    loss['feature_distill_loss'] = self.loss_feature_distill_alpha * feature_distill_loss
                elif (self.loss_feature_type == 'gussian mask L2'):
                    feature_teacher = self.teacher.extract_feat(inputs)[0]
                    feature_student = self.student.extract_feat(inputs_fmri)[0]
                    feature_student_mapped = self.feature_channel_mapper((feature_student,))[0]

                    feature_distill_loss = torch.mean((gm.unsqueeze(1) * ((feature_teacher - feature_student_mapped) ** 2))) * self.loss_feature_distill_alpha

                    loss['feature_distill_loss'] = feature_distill_loss
                elif (self.loss_feature_type == 'gussian mask L1'):
                    feature_teacher = self.teacher.extract_feat(inputs)[0]
                    feature_student = self.student.extract_feat(inputs_fmri)[0]
                    feature_student_mapped = self.feature_channel_mapper((feature_student,))[0]

                    feature_distill_loss = torch.mean((gm.unsqueeze(1) * (torch.abs(feature_teacher - feature_student_mapped)))) * self.loss_feature_distill_alpha

                    loss['feature_distill_loss'] = feature_distill_loss

            # ----------------------------- encoded_feat_loss ---------------------------- #
            if (self.loss_encoded_feature_distill_alpha > 1e-9):
                encoder_inputs_dict_t, decoder_inputs_dict_t = self.teacher.pre_transformer(
                    (feature_teacher,), data_samples)
                encoder_outputs_dict_t = self.teacher.forward_encoder(**encoder_inputs_dict_t)
                encoder_inputs_dict_s, decoder_inputs_dict_s = self.student.pre_transformer(
                    (feature_student,), data_samples)
                encoder_outputs_dict_s = self.student.forward_encoder(**encoder_inputs_dict_s)
                encoded_feature_distill_loss = F.mse_loss(encoder_outputs_dict_t['memory'], encoder_outputs_dict_s['memory'])
                loss['encoded_feature_distill_loss'] = self.loss_encoded_feature_distill_alpha * encoded_feature_distill_loss

            return loss
        elif mode == 'predict':
            return self.student.predict(inputs_fmri, data_samples)
        elif mode == 'tensor':
            return self.student._forward(inputs_fmri, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')


@MODELS.register_module()
class DETR_1D(DETR_upgradation):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding1D(
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

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such as
                `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """

        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        batch_size, feat_dim, _ = feat.shape
        # construct binary masks which for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            masks = None
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks, input=feat)
        else:
            masks = feat.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero values represent
            # ignored positions, while zero values mean valid positions.

            masks = F.interpolate(
                masks.unsqueeze(1),
                size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, l] -> [bs, l, c]
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        # [bs, l] -> [bs, l]
        if masks is not None:
            masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        encoder_inputs_dict = dict(
            feat=feat, feat_mask=masks, feat_pos=pos_embed)
        decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict
