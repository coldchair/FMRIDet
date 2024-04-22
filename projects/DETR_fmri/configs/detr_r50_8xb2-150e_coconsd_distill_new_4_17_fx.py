_base_ = [
    # '../../configs/_base_/datasets/coconsd_detection.py',
    './dataset.py',
    '../../../configs/_base_/default_runtime.py',
    './detr_config_student_ch64.py',
    './detr_config_teacher.py'
]

student_config = {{_base_.student_config}}
teacher_config = {{_base_.teacher_config}}

model = dict(
    type='DETR_distill_new',
    teacher_cfg = teacher_config,
    student_cfg = student_config,
    loss_feature_distill_alpha = 2.0,
    feature_distill_size = 64,
    loss_encoded_feature_distill_alpha = 0.0,
    loss_feature_type = 'gussian mask L2',
    data_preprocessor=dict(
        type='DetDataPreprocessor_fmri',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# learning policy
max_epochs = 130
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

vis_backends = [dict(type='TensorboardVisBackend'), dict(type = 'LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
custom_hooks = [dict(type='WriteValidationLossHook')]

resume = True

# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'