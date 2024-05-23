# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmdet.models.layers import MLP, inverse_sigmoid
from mmdet.models.dense_heads.conditional_detr_head import ConditionalDETRHead
from mmdet.models.dense_heads.dab_detr_head import DABDETRHead
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean)
from mmdet.models.utils import multi_apply
from mmdet.models.task_modules import build_assigner
from mmdet.models.losses import QualityFocalLoss
from mmcv.ops import batched_nms

@MODELS.register_module()
class DABDETRHead_distill(DABDETRHead):
    def __init__(self,
                 distill_assigner : dict = None,
                 loss_cls_distill: dict = None,
                 loss_bbox_distill: dict = None,
                 loss_iou_distill: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_query = 300
        
        if (loss_cls_distill != None) and (loss_bbox_distill != None) and (loss_iou_distill != None):
            self.loss_cls_distill = MODELS.build(loss_cls_distill)
            self.loss_bbox_distill = MODELS.build(loss_bbox_distill)
            self.loss_iou_distill = MODELS.build(loss_iou_distill)

        if kwargs.get('train_cfg', None) != None:
            if (kwargs['train_cfg'].get('distill_assigner', None) != None):
                distill_assigner = kwargs['train_cfg']['distill_assigner']
                self.distill_assigner = TASK_UTILS.build(distill_assigner)
        
        self.sum_time = 0

    # 加入 nms
    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        score_thr = self.test_cfg.get('score_thr', 0)
        with_nms = self.test_cfg.get('nms', None)

        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        if score_thr > 0:
            valid_mask = scores > score_thr
            scores = scores[valid_mask]
            bbox_pred = bbox_pred[valid_mask]
            det_labels = det_labels[valid_mask]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels,
                                                self.test_cfg.nms)
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[:max_per_img]

        return results

    def distill_loss(self, hidden_states: Tensor, references: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            hidden_states (Tensor): Features from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, dim).
            references (Tensor): References from the transformer decoder, has
               shape (num_decoder_layers, bs, num_queries, 2).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
        outs = self(hidden_states, references)
        """
            outs : 
            - layers_cls_scores (Tensor): Outputs from the classification head,
              shape (num_decoder_layers, bs, num_queries, cls_out_channels).
              Note cls_out_channels should include background.
            - layers_bbox_preds (Tensor): Sigmoid outputs from the regression
              head with normalized coordinate format (cx, cy, w, h), has shape
              (num_decoder_layers, bs, num_queries, 4).
        """
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses, labels_list = self.loss_by_feat_v2(*loss_inputs)
        return losses, loss_inputs, labels_list
    
    # v2 : return losses and labels_list
    def loss_by_feat_v2(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou, labels_list = multi_apply(
            self.loss_by_feat_single_v2,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict, labels_list

    def loss_by_feat_single_v2(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores),
                label_weights,
                avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_iou, labels_list
    
    def loss_by_feat_distill(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        mask_feats: Tensor, # shape : num_layers, bs, num_queries
        all_layers_cls_scores_t: Tensor,
        all_layers_bbox_preds_t: Tensor,
        mask_feats_t: Tensor,
        batch_gt_instances: InstanceList, # 实际上，这里是 teacher model 的输出
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single_distill,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            mask_feats,
            all_layers_cls_scores_t,
            all_layers_bbox_preds_t,
            mask_feats_t,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls_distill'] = losses_cls[-1]
        loss_dict['loss_bbox_distill'] = losses_bbox[-1]
        loss_dict['loss_iou_distill'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_distill'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_distill'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou_distill'] = loss_iou_i
            num_dec_layer += 1
        # print(self.sum_time)
        return loss_dict

    def loss_by_feat_single_distill(self,
                            cls_scores: Tensor, bbox_preds: Tensor, mask_feats: Tensor,
                            cls_scores_t: Tensor, bbox_preds_t: Tensor, mask_feats_t: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        # cls_scores : shape : bs * num_queries * num_classes
        # print(batch_gt_instances[0].bboxes)
        # print(bbox_preds_t[0])

        batch_gt_instances = [
            InstanceData(metainfo=batch_gt_instances[i].metainfo,
                        bboxes=bbox_preds_t[i][mask_feats_t[i]],
                        labels=cls_scores_t[i][mask_feats_t[i]])
            for i in range(len(batch_gt_instances))
        ]
        # ---------------------------------- modify ---------------------------------- #

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        mask_feats_list = [mask_feats[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets_distill(
            cls_scores_list, bbox_preds_list, mask_feats_list,
            batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # 由于 distill 的时候 正例/负例数量对于每个样本都是不一样的，所以需要分开计算

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # ---------------------------------- modify ---------------------------------- #
        # label_weights = label_weights.unsqueeze(-1).repeat(1, self.num_classes)

        # batch_size
        bs = cls_scores.size(0)

        loss_cls = torch.sum(
            torch.stack(
                [
                    self.loss_cls_distill(
                        cls_scores[i][mask_feats[i]],
                        labels_list[i],
                        label_weights_list[i].unsqueeze(-1).repeat(1, self.num_classes),
                        avg_factor = cls_avg_factor
                    ) for i in range(bs)
                ]
            )
        )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for i, (img_meta, bbox_pred) in enumerate(zip(batch_img_metas, bbox_preds)):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred[mask_feats_list[i]].size(0), 1)
            factors.append(factor)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bboxes = []
        bboxes_gt = []
        for i in range(bs):
            bboxes.append(bbox_cxcywh_to_xyxy(bbox_preds_list[i][mask_feats[i]]) * factors[i])
            bboxes_gt.append(bbox_cxcywh_to_xyxy(bbox_targets_list[i]) * factors[i])

        # regression IoU loss, defaultly GIoU loss
        loss_iou = torch.sum(
            torch.stack(
                [
                    self.loss_iou_distill(bboxes[i],
                                        bboxes_gt[i],
                                        bbox_weights_list[i],
                                        avg_factor=num_total_pos)
                    for i in range(bs)
                ]
            )
        )

        # regression L1 loss
        loss_bbox = torch.sum(
            torch.stack(
                [
                    self.loss_bbox_distill(bbox_preds_list[i][mask_feats[i]],
                                        bbox_targets_list[i],
                                        bbox_weights_list[i],
                                        avg_factor=num_total_pos)
                    for i in range(bs)
                ]
            )
        )

        return loss_cls, loss_bbox, loss_iou

    def get_targets_distill(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    mask_feats_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_targets_single_distill,
                                      cls_scores_list, bbox_preds_list,
                                      mask_feats_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
        

    def _get_targets_single_distill(self, cls_score: Tensor, bbox_pred: Tensor,
                            mask_feats: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        
        cls_score = cls_score[mask_feats]
        bbox_pred = bbox_pred[mask_feats]

        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        # ---------------------------------- modify ---------------------------------- #
        gt_instances.bboxes = bbox_cxcywh_to_xyxy(gt_instances.bboxes) * factor
        # ---------------------------------- modify ---------------------------------- #


        # import time
        # s1 = time.time()
        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.distill_assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        # self.sum_time += time.time() - s1

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        # labels = gt_bboxes.new_full((num_bboxes, ),
        #                             self.num_classes,
        #                             dtype=torch.long)
        # ---------------------------------- modify ---------------------------------- #
        labels = gt_bboxes.new_full((num_bboxes, self.num_classes),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)