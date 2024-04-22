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
        if hasattr(runner.model, 'val_loss_step'):
            outputs_loss = runner.model.val_loss_step(data_batch)
        else:
            outputs_loss = runner.model.module.val_loss_step(data_batch)
        for key, value in outputs_loss.items():
            runner.message_hub.update_scalar(f'test/{key}_test', value.item())
        