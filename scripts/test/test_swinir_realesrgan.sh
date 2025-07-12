# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"

# python train.py --config options/train/train_lcm_sr.yml

# Experiment target
# 测试scale distillation的效果，先尝试用这个思路train seesr

experiments=(
8gpus_bs8_train_fm_transformer_x4_LSDIR_c512_s384_q3dot5_ffhq_10k_256_400k_lr5e-4_latent_intep_looformer_p1_std0dot1_our_vae_ema_f8c32_nona_swinir
)

dataset_names=(
# DIV2K
# DIV2K_patch3k
# realsr_48
RealSR
# DRealSR
# DPED
# RealLR200
)

lq_dataset_dirs=(
# /home/sist/luoxin/datasets/DIV2K/RealESRGAN/LR/X4
# /home/sist/luoxin/datasets/real_world_sr_testset/LR/X4
/home/sist/luoxin/datasets/RealSR/RealSR\(V3\)_2/Test/LR/4
# /home/sist/luoxin/datasets/DRealSR/Test_x4/test_LR
# /home/sist/luoxin/datasets/DPED_test
# /home/sist/luoxin/datasets/RealLR200
)

gt_dataset_dirs=(
# /home/sist/luoxin/datasets/DIV2K/RealESRGAN/HR/X4
# /home/sist/luoxin/datasets/real_world_sr_testset/LR/X4
/home/sist/luoxin/datasets/RealSR/RealSR\(V3\)_2/Test/HR/4
# /home/sist/luoxin/datasets/DRealSR/Test_x4/test_HR
# None
# None
)

post_fixs=(
# ''
# ''
'LR4'
# 'x1'
# ''
# ''
)

gt_post_fixs=(
# ''
# ''
'HR'
# 'x4'
# ''
''
)

for((i=0;i<${#experiments[@]};i++))
do
    for((j=0;j<${#dataset_names[@]};j++))
    do
        # conda activate py3.10+pytorch2.4+cu121
        # accelerate launch --multi_gpu \
        # --main_process_port 41017 \
        # --num_machines 1 \
        # --mixed_precision no \
        # --dynamo_backend no \
        # test_sr.py --experiment_name ${experiments[i]} \
        # --dataset_names ${dataset_names[j]} \
        # --dataset_dirs ${lq_dataset_dirs[j]} \
        # --pre_upsample \
        # --test_posterior_mean \
        # --result_dir full_results_posterior_mean

        conda activate pyiqa
        accelerate launch --multi_gpu \
        --main_process_port 41017 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        measure_metrics.py --experiment_name ${experiments[i]} \
        --dataset_names ${dataset_names[j]} \
        --dataset_dirs ${gt_dataset_dirs[j]} \
        --post_fix "${post_fixs[i]}_sr" \
        --gt_post_fix "${gt_post_fixs[i]}" \
        --sub_img_dir visualization \
        --result_dir full_results_posterior_mean
    done
done