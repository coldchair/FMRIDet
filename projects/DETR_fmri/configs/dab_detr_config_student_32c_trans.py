_base_ = ['dab_detr_config_student.py']

student_config = dict(
    backbone=dict(
        type='Backbone_fmri_transformer',
        output_dim = (32, 25, 25),
        input_dim = 26688,
        middle_dim = 4096 * 2,
    ),
)