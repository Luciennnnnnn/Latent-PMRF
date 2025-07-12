# # conda create -n py3.10+cu118 python=3.10
# conda create -n py3.10+cu121 python=3.10

# # pip install torch==2.1.2 torchvision torchaudio xformers --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118
# pip install torch==2.4.1 torchvision torchaudio xformers --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121

packages=(
opencv-python-headless
scipy==1.11.1
imageio
rawpy
)

for((i=0;i<${#packages[@]};i++))
do
    pip install ${packages[i]} -i https://pypi.tuna.tsinghua.edu.cn/simple
done

packages=(
diffusers[torch]
accelerate
datasets
transformers
peft
loralib
fairscale
einops
timm
)

for((i=0;i<${#packages[@]};i++))
do
    pip install ${packages[i]} -i https://pypi.tuna.tsinghua.edu.cn/simple
done

packages=(
wandb
matplotlib
omegaconf
fvcore
pyiqa
python-dotenv
)

for((i=0;i<${#packages[@]};i++))
do
    pip install ${packages[i]} -i https://pypi.tuna.tsinghua.edu.cn/simple
done