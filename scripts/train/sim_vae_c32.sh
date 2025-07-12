# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

cd ../

experiments=(
sim_vae_c32
)

for((i=0;i<${#experiments[@]};i++))
do
    accelerate launch --multi_gpu \
    --main_process_port 45903 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    train_vae.py --config options/train/${experiments[i]}.yml
done