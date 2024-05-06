# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import mmengine.fileio as fileio
import numpy as np

import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS

@TRANSFORMS.register_module()
class LoadfMRIFromFile(BaseTransform):
    """Load an fmri from file.
    """
    def __init__(self,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        filename = results['fmri_path']
        a = np.load(filename).astype('float32')
        if ('padding_zeros' in results):
            p = results['padding_zeros']
            if (p != None):
                # print(a.shape)
                a = np.pad(a, [p,])
                # print(a.shape)
                # print(filename)
        results['fmri'] = a
        return results