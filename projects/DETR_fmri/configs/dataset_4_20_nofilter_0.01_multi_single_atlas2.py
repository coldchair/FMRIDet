_base_ = [
    'dataset_4_20_nofilter_0.01_multi_single.py',
]
dataset_type = 'CocoNSDDataset'
data_root = './'
image_dir = '../nsd_processed_data/all_images'

# default : each sample

SAVE_ROOT_DIR = '../nsd_processed_data'
subj_list = ['subj01', 'subj02', 'subj05', 'subj07']

fmri_prefix_name = []

for i, subj in enumerate(subj_list):
    fmri_prefix_name.append(f'{SAVE_ROOT_DIR}/mrifeat/{subj}/nsdgeneral')

input_size = [15724, 14278, 13039, 12682]
input_size_sum = sum(input_size) # 55723

train_dataloader = dict(
    dataset=dict(
        input_size = input_size,
        fmri_prefix_name = fmri_prefix_name,
    )
)
val_dataloader = dict(
    dataset=dict(
        input_size = input_size,
        fmri_prefix_name = fmri_prefix_name,
    )
)
test_dataloader = dict(
    dataset=dict(
        input_size = input_size,
        fmri_prefix_name = fmri_prefix_name,
    )
)