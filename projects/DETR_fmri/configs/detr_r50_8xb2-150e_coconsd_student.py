_base_ = [
    # '../../configs/_base_/datasets/coconsd_detection.py',
    './dataset.py',
    '../../../configs/_base_/default_runtime.py',
    './detr_config_student.py',
]

model = {{_base_[2].student_config}}

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# learning policy
max_epochs = 120
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[110],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

vis_backends = [dict(type='TensorboardVisBackend'), dict(type = 'LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
    
# load_from = '/mnt/workspace/maxinzhu/denghan/FMRIDet/work_dirs/detr_r50_8xb2-150e_coconsd_student_1/epoch_20.pth'
# load_from = '/mnt/workspace/maxinzhu/denghan/FMRIDet/work_dirs/detr/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'