# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    './dataset_4_20_nofilter_0.01_multi_single_atlas2.py',
]

SAVE_ROOT_DIR = '../nsd_processed_data'
ann_file = f'{SAVE_ROOT_DIR}/instances_0_73000_0.001.json'

subj_list = ['subj01', 'subj02', 'subj05', 'subj07']
index_file_te = []
for i, subj in enumerate(subj_list):
    index_file_te.append( f'{SAVE_ROOT_DIR}/mrifeat/{subj}/index_each_tr_50.npy')

index_file_te = [index_file_te[0], [], [], []]

train_dataloader = dict(
    dataset = dict(
        ann_file = ann_file
    )
)
val_dataloader = dict(
    dataset = dict(
        ann_file = ann_file,
        fmri_suffix_name = 'tr_',
        index_file = index_file_te,
    )
)
test_dataloader = dict(
    dataset = dict(
        ann_file = ann_file,
        fmri_suffix_name = 'tr_',
        index_file = index_file_te,
    )
)
