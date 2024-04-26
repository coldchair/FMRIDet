# Copyright (c) OpenMMLab. All rights reserved.
# train_data with argumentation(mixup)
custom_imports = dict(
    imports=['projects.DETR_fmri.codetr'], allow_failed_imports=False)

dataset_type = 'CocoNSDDataset'
data_root = './'
image_dir = '../nsd_processed_data/all_images'

# default : each sample

subj = 'subj01'
SAVE_ROOT_DIR = '../nsd_processed_data'
ann_file = f'{SAVE_ROOT_DIR}/instances_0_73000_0.1_filter_cls.json'
index_file_tr = f'{SAVE_ROOT_DIR}/mrifeat/{subj}/index_each_tr_ex.npy'
index_file_te = f'{SAVE_ROOT_DIR}/mrifeat/{subj}/index_each_te.npy'

# early ventral midventral midlateral lateral parietal
rois = ['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal']

fmri_files_path_tr = []
fmri_files_path_te = []

for roi in rois:
    fmri_files_path_tr.append(f'{SAVE_ROOT_DIR}/mrifeat/{subj}/{subj}_{roi}_betas_tr_ex.npy')
    fmri_files_path_te.append(f'{SAVE_ROOT_DIR}/mrifeat/{subj}/{subj}_{roi}_betas_te.npy')

dataloader_type = 'DetDataLoader_fmri'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs_fmri')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs_fmri',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        index_file = index_file_tr,
        fmri_files_path = fmri_files_path_tr,
        # input_dim = 2,
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img=image_dir),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        index_file = index_file_te,
        fmri_files_path = fmri_files_path_te,
        # input_dim = 2,
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img=image_dir),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric_modified',
    ann_file=ann_file,
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
