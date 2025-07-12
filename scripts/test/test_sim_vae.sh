# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"

# python train.py --config options/train/train_lcm_sr.yml

# Experiment target
# 测试scale distillation的效果，先尝试用这个思路train seesr

experiments=(
8gpus_bs8_train_vae_LSDIR_c512_s384_q3dot5_ffhq10k_150k_lr1e-4_our_vae_f8c32_patch256_vgg_loss_weight0dot05
)

dataset_names=(
# LSDIR_c512_s384_q3dot5
CelebA
)

dataset_dirs=(
# /home/sist/luoxin/datasets/LSDIR/train/HR_sub_c512_s384
/home/sist/luoxin/datasets/face/celeba_512_validation/gt
)

file_paths=(
# /home/sist/luoxin/datasets/LSDIR/train/HR_sub_c512_s384_random_order_image_paths.json
None
)

num_samples=(
1000
1000
)

for((i=0;i<${#experiments[@]};i++))
do
    for((j=0;j<${#dataset_names[@]};j++))
    do
        conda activate py3.10+pytorch2.4+cu121
        accelerate launch --multi_gpu \
        --main_process_port 46501 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        test_vae.py --experiment_name ${experiments[i]} \
        --dataset_names ${dataset_names[j]}_${num_samples[j]} \
        --dataset_dirs ${dataset_dirs[j]} \
        --file_paths ${file_paths[j]} \
        --num_samples ${num_samples[j]} \
        --test_ema \
        --result_dir experiments/${experiments[i]}/full_results

        # conda activate pyiqa
        # accelerate launch --multi_gpu \
        # --main_process_port 46501 \
        # --num_machines 1 \
        # --mixed_precision no \
        # --dynamo_backend no \
        # measure_metrics.py --experiment_name ${experiments[i]} \
        # --dataset_names ${dataset_names[j]}_${num_samples[j]} \
        # --dataset_dirs ${dataset_dirs[j]} \
        # --post_fix _pred \
        # --result_dir full_results
    done
done