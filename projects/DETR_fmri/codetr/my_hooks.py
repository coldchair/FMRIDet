from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

from typing import Dict, Optional, Sequence, Union

from mmengine import is_method_overridden

DATA_BATCH = Optional[Union[dict, tuple, list]]

from collections import OrderedDict

@HOOKS.register_module()
class WriteValidationLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval: int = 50) -> None:
        self.interval = interval
    
    def before_run(self, runner) -> None:
        self.json_log_path = f'{runner.timestamp}.json'

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        if hasattr(runner.model, 'val_loss_step'):
            outputs_loss = runner.model.val_loss_step(data_batch)
        else:
            outputs_loss = runner.model.module.val_loss_step(data_batch)
        for key, value in outputs_loss.items():
            runner.message_hub.update_scalar(f'val/{key}_val', value.item())

    def after_test_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        # data_batch.data_samples : list of DetDataSample
        '''
            <DetDataSample(

        META INFORMATION
        img_path: '/home/bingxing2/ailab/group/ai4bio/public/nsd_processed_data/all_images/005602.png'
        img_shape: (800, 800)
        ori_shape: (425, 425)
        scale_factor: (1.8823529411764706, 1.8823529411764706)
        img_id: 5602

        DATA FIELDS
        ignored_instances: <InstanceData(
            
                META INFORMATION
            
                DATA FIELDS
                bboxes: HorizontalBoxes(
                    tensor([], size=(0, 4)))
                labels: tensor([], dtype=torch.int64)
            ) at 0x4004d5cb40a0>
        gt_instances: <InstanceData(
            
                META INFORMATION
            
                DATA FIELDS
                bboxes: HorizontalBoxes(
                    tensor([[  0.0000,  63.1926,  73.9918, 140.5187]]))
                labels: tensor([11])
            ) at 0x4004d5cb4400>
    ) at 0x4004d5cb43d0>]}
        '''
        if hasattr(runner.model, 'val_loss_step'):
            outputs_loss = runner.model.val_loss_step(data_batch)

            # t0, t1, t2 = runner.model.teacher.test_feature_map(data_batch)
            # s0, s1, s2 = runner.model.student.test_feature_map(data_batch)
            # s0, s1, s2 = runner.model.test_feature_map(data_batch)

            # # t0 : bs * 2048 * 25 * 25
            # # t1 : bs * 256 * 25 * 25
            # # t2 : bs * 625 * 256

            # # t0 = t0.cpu().numpy()
            # # t1 = t1.cpu().numpy()
            # # t2 = t2.cpu().permute(0, 2, 1).reshape(-1, 256, 25, 25).numpy()
            # s0 = s0.cpu().numpy()
            # s1 = s1.cpu().numpy()
            # s2 = s2.cpu().permute(0, 2, 1).reshape(-1, 64, 25, 25).numpy()

            # # print(t0.shape)
            # # print(t1.shape)
            # # print(t2.shape)
            # # print(s0.shape)
            # # print(s1.shape)
            # # print(s2.shape)

            # # print(data_batch)
            # n = len(data_batch['data_samples'])


            # for i in range(n):
            #     # print(data_batch['data_samples'][i].metainfo_values)
            #     id = data_batch['data_samples'][i].img_id
            #     print(f'{id:06}')

            #     import numpy as np
            #     save_dir = '/home/bingxing2/ailab/scx7kzd/denghan/FMRIDet/code/feature_no_distill/'
            #     # np.save(f'{save_dir}{id:06}_t0.npy', t0[i])
            #     # np.save(f'{save_dir}{id:06}_t1.npy', t1[i])
            #     # np.save(f'{save_dir}{id:06}_t2.npy', t2[i])
            #     np.save(f'{save_dir}{id:06}_s0.npy', s0[i])
            #     np.save(f'{save_dir}{id:06}_s1.npy', s1[i])
            #     np.save(f'{save_dir}{id:06}_s2.npy', s2[i])


            # print(data_batch)
        else:
            outputs_loss = runner.model.module.val_loss_step(data_batch)
        for key, value in outputs_loss.items():
            runner.message_hub.update_scalar(f'test/{key}_test', value.item())
        