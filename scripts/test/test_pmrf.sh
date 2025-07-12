# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"

# python train.py --config options/train/train_lcm_sr.yml

# Experiment target
# 测试scale distillation的效果，先尝试用这个思路train seesr

dataset_names=(
celeba_512_validation
)

dataset_dirs=(
/home/sist/luoxin/datasets/face/celeba_512_validation/gt
)

shifts=(
0.25
# 0.5
# 1.0
# 1.5
# 2.0
# 2.5
# 3.0
)

# for((i=0;i<${#shifts[@]};i++))
# do
#     for((j=0;j<${#dataset_names[@]};j++))
#     do
#         conda activate pyiqa
#         accelerate launch --multi_gpu \
#         --main_process_port 41360 \
#         --num_machines 1 \
#         --mixed_precision no \
#         --dynamo_backend no \
#         measure_metrics.py --base_dir /home/sist/luoxin/projects/PMRF/results \
#         --dataset_names ${dataset_names[j]}_${num_samples[j]} \
#         --dataset_dirs ${dataset_dirs[j]} \
#         --post_fix "" \
#         --sub_img_dir restored_images_posterior_mean \
#         --result_dir rec_shift${shifts[i]}
#     done
# done

# conda activate pyiqa
# accelerate launch --multi_gpu \
# --main_process_port 41360 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# measure_metrics.py --base_dir /home/sist/luoxin/projects/PMRF/results \
# --dataset_names ${dataset_names[j]}_${num_samples[j]} \
# --dataset_dirs ${dataset_dirs[j]} \
# --post_fix "" \
# --sub_img_dir restored_images_posterior_mean \
# --result_dir rec

# conda activate pyiqa
# accelerate launch --multi_gpu \
# --main_process_port 41360 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# measure_metrics.py --base_dir /home/sist/luoxin/projects/PMRF/results \
# --dataset_names ${dataset_names[j]}_${num_samples[j]} \
# --dataset_dirs ${dataset_dirs[j]} \
# --post_fix "" \
# --sub_img_dir restored_images \
# --result_dir rec

# conda activate pyiqa
# accelerate launch --multi_gpu \
# --main_process_port 41360 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# measure_metrics.py --base_dir /home/sist/luoxin/projects/PMRF/results \
# --dataset_names ${dataset_names[j]}_${num_samples[j]} \
# --dataset_dirs ${dataset_dirs[j]} \
# --post_fix "" \
# --sub_img_dir restored_images \
# --result_dir bs32_200k_shift1.0_steps25

# conda activate pyiqa
# accelerate launch --multi_gpu \
# --main_process_port 41360 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# measure_metrics.py --base_dir /home/sist/luoxin/projects/PMRF/results \
# --dataset_names ${dataset_names[j]}_${num_samples[j]} \
# --dataset_dirs ${dataset_dirs[j]} \
# --post_fix "" \
# --sub_img_dir restored_images \
# --result_dir bf16_bs32_200k_normalize_isbdduck_shift1.0_steps25

# conda activate pyiqa
# accelerate launch --multi_gpu \
# --main_process_port 41360 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# measure_metrics.py --base_dir /home/sist/luoxin/projects/PMRF/results \
# --dataset_names ${dataset_names[j]}_${num_samples[j]} \
# --dataset_dirs ${dataset_dirs[j]} \
# --post_fix "" \
# --sub_img_dir restored_images \
# --result_dir bf16_bs32_200k_normalize_noise_std_0.14142_j5unr1u8_shift1.0_steps25

conda activate pyiqa
accelerate launch --multi_gpu \
--main_process_port 41360 \
--num_machines 1 \
--mixed_precision no \
--dynamo_backend no \
measure_metrics.py --base_dir /home/sist/luoxin/projects/PMRF/results \
--dataset_names ${dataset_names[j]}_${num_samples[j]} \
--dataset_dirs ${dataset_dirs[j]} \
--post_fix "" \
--sub_img_dir restored_images \
--result_dir bf16_bs32_200k_normalize_noise_std_0.2_zofelfrn_shift1.0_steps25