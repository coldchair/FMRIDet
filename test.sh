for i in $(seq 20 20)
do
python ./tools/test.py \
    /mnt/workspace/maxinzhu/denghan/FMRIDet/projects/DETR_fmri/configs/detr_r50_8xb2-150e_coconsd_distill_4_17_fx_filter_cls_mixup_100q_ave.py \
    /mnt/workspace/maxinzhu/denghan/FMRIDet/work_dirs/detr_r50_8xb2-150e_coconsd_distill_4_17_fx_filter_cls_mixup_100q_ave/epoch_20.pth \
    --out results_4_25/results_${i}.pkl --show-dir my_test
done