# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    './dataset_4_20_nofilter_0.01_multi_single_atlas2.py',
]

SAVE_ROOT_DIR = '../nsd_processed_data'
ann_file = f'{SAVE_ROOT_DIR}/instances_0_73000_0.001.json'

train_dataloader = dict(
    dataset = dict(
        ann_file = ann_file
    )
)
val_dataloader = dict(
    dataset = dict(
        ann_file = ann_file
    )
)
test_dataloader = dict(
    dataset = dict(
        ann_file = ann_file
    )
)
