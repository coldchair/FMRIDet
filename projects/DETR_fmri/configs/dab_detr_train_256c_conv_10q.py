_base_ = [
    # '../../configs/_base_/datasets/coconsd_detection.py',
    './dataset_4_20_nofilter_0.01.py',
    # './dataset_train50.py',
    '../../../configs/_base_/default_runtime.py',
    './dab_detr_config_student_256c_conv.py',
]

student_config = _base_.student_config

student_config.num_queries = 10
student_config.test_cfg = dict(max_per_img=10)

model = student_config

train_dataloader = dict(
    batch_size = 12,
    dataset = dict(input_dim = 1,
                   padding_zeros = 32)
)
val_dataloader = dict(
    batch_size = 12,
    dataset = dict(input_dim = 1,
                   padding_zeros = 32)
)
test_dataloader = dict(
    batch_size = 12,
    dataset = dict(input_dim = 1,
                   padding_zeros = 32)
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
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# learning policy
max_epochs = 150
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
        milestones=[100],
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