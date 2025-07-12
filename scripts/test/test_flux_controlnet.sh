# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

cd ../

dataset_names=(
DIV2K
RealSR
# DRealSR
DPED
RealLR200
RealPhoto60
)

datasets=(
/home/sist/luoxin/datasets/DIV2K/RealESRGAN/LR/X4
/home/sist/luoxin/datasets/RealSR/RealSR\(V3\)_2/Test/4
# /home/sist/luoxin/datasets/DRealSR/Test_x4/test_LR
/home/sist/luoxin/datasets/DPED_test
/home/sist/luoxin/datasets/RealLR200
/home/sist/luoxin/datasets/RealPhoto60
)

inference_steps=(
# 1
# 2
# 4
# 8
# 16
# 32
# 50
28
)

guidance_scales=(
# 2.2
3.5
# 7.0
# 3.0
)

controlnet_conditioning_scales=(
0.8
0.9
1.0
)

for((i=0;i<${#datasets[@]};i++))
do
    for((j=0;j<${#inference_steps[@]};j++))
    do
        for((k=0;k<${#guidance_scales[@]};k++))
        do
            for((l=0;l<${#controlnet_conditioning_scales[@]};l++))
            do
                accelerate launch --multi_gpu \
                --main_process_port 45234 \
                --num_machines 1 \
                --mixed_precision no \
                --dynamo_backend no \
                test_sr_flux_controlnet.py \
                --input_dir ${datasets[i]} \
                --output_dir /home/sist/luoxin/projects/LCM_SR/results/${dataset_names[i]}/FLUX/DDPMScheduler/${inference_steps[j]}_${guidance_scales[k]}_${controlnet_conditioning_scales[l]} \
                --num_inference_steps ${inference_steps[j]} \
                --guidance_scale ${guidance_scales[k]} \
                --controlnet_conditioning_scale ${controlnet_conditioning_scales[l]}
            done
        done
    done
done


# pre_down_scales=(
# # 1
# # 0.8
# # 0.5
# 0.25
# )

# for((i=0;i<${#pre_down_scales[@]};i++))
# do
#     accelerate launch --multi_gpu \
#     --main_process_port 46711 \
#     --num_machines 1 \
#     --mixed_precision no \
#     --dynamo_backend no \
#     test_seesr.py \
#     --pretrained_model_path stabilityai/stable-diffusion-2-base \
#     --seesr_model_path pretrained_models/seesr \
#     --ram_ft_path pretrained_models/ram/DAPE.pth \
#     --image_path /home/sist/luoxin/datasets/LSDIR/train/HR_sub_c768_s384 \
#     --output_dir /home/sist/luoxin/datasets/LSDIR/train/HR_sub_c768_s384_seesr_pre_down_scale_${pre_down_scales[i]} \
#     --start_point lr \
#     --num_inference_steps 50 \
#     --guidance_scale 5.5 \
#     --process_size 512 \
#     --pre_down_scale ${pre_down_scales[i]} \
#     --upscale 1
# done