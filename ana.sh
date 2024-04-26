for i in $(seq 11 17)
do
    python tools/analysis_tools/analyze_results.py \
            /mnt/workspace/maxinzhu/denghan/FMRIDet/projects/DETR_fmri/configs/detr_r50_8xb2-150e_coconsd_distill_4_17_fx_mixup_100q_ave.py \
        results/results_${i}.pkl \
        results/results_${i} \
        --show-score-thr 0.1 --topk 100
done