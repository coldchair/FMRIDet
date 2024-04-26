_base_ = [
    # '../../configs/_base_/datasets/coconsd_detection.py',
    './dataset_4_20_mixup_nofilter_0.01.py',
    # './dataset_train50.py',
    '../../../configs/_base_/default_runtime.py',
    './dab_detr_config_student_32c_trans.py',
    './dab_detr_config_teacher.py'
]

student_config = {{_base_.student_config}}
teacher_config = {{_base_.teacher_config}}

model = dict(
    type='DABDETR_distill',
    teacher_cfg = teacher_config,
    student_cfg = student_config,
    loss_feature_distill_alpha = 2.0,
    loss_encoded_feature_distill_alpha = 0.0,
    loss_feature_type = 'gussian mask L2',
    freeze_student_decoder_bool = True,
    data_preprocessor=dict(
        type='DetDataPreprocessor_fmri',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='PackDetInputs')
# ]
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# learning policy
max_epochs = 50
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
        milestones=[40],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enable=False)

vis_backends = [dict(type='TensorboardVisBackend'), dict(type = 'LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    )
custom_hooks = [dict(type='WriteValidationLossHook')]

resume = True