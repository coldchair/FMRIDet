# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    './dataset_4_20_nofilter_0.001_multi_single_atlas2.py',
]

image_dir = '/home/bingxing2/ailab/group/ai4bio/public/nsd_processed_data/all_images/'
# image_dir = '/home/bingxing2/ailab/scx7kzd/denghan/fMRI-reconstruction-NSD/img_rec_s1'
# image_dir = '/home/bingxing2/ailab/scx7kzd/denghan/takagi/subj01/samples'


test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img=image_dir),
    ),
)