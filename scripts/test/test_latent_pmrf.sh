# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"

# python train.py --config options/train/train_lcm_sr.yml

# Experiment target
# 测试scale distillation的效果，先尝试用这个思路train seesr

experiments=(
8gpus_bs8_train_fm_transformer_x4_FFHQ_512_400k_lr5e-4_latent_intep_looformer_plain_2n_4n_6g_dwconv_patch_conv_p1_std0dot1_our_vae_ema_f8c32_nona
)

dataset_names=(
CelebA
)

lq_dataset_dirs=(
/home/sist/luoxin/datasets/face/celeba_512_validation/lq
)

gt_dataset_dirs=(
/home/sist/luoxin/datasets/face/celeba_512_validation/gt
)

num_samples=(
3000
)

shift=(
# 0.25
# 0.5
1.0
# 1.5
# 2.0
# 2.5
# 3.0
)

for((i=0;i<${#experiments[@]};i++))
do
    for((j=0;j<${#dataset_names[@]};j++))
    do
        for((k=0;k<${#shift[@]};k++))
        do
            conda activate py3.10+pytorch2.4+cu121
            accelerate launch --multi_gpu \
            --main_process_port 42000 \
            --num_machines 1 \
            --mixed_precision no \
            --dynamo_backend no \
            test_sr_fm_pixel.py --experiment_name ${experiments[i]} \
            --dataset_names ${dataset_names[j]}_${num_samples[j]} \
            --dataset_dirs ${lq_dataset_dirs[j]} \
            --num_samples ${num_samples[j]} \
            --num_inference_step 25 \
            --test_ema \
            --result_dir full_results_ema_shift${shift[k]}

            # conda activate pyiqa
            # accelerate launch --multi_gpu \
            # --main_process_port 42000 \
            # --num_machines 1 \
            # --mixed_precision no \
            # --dynamo_backend no \
            # measure_metrics.py --experiment_name ${experiments[i]} \
            # --dataset_names ${dataset_names[j]}_${num_samples[j]} \
            # --dataset_dirs ${gt_dataset_dirs[j]} \
            # --post_fix _sr \
            # --sub_img_dir visualization \
            # --result_dir full_results_ema_shift${shift[k]}/FlowMatchEulerDiscreteScheduler/25
        done
    done
done