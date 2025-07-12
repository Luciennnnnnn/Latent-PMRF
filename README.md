<p align="center">
  <img src="assets/brand.jpeg" width="65%">
</p>

<p align="center">
  <!-- <a href="https://vectorspacelab.github.io/OmniGen2"><img src="https://img.shields.io/badge/Project%20Page-OmniGen2-yellow" alt="project page"></a> -->
  <a href="https://arxiv.org/abs/2507.00447"><img src="https://img.shields.io/badge/arXiv%20paper-2507.00447-b31b1b.svg" alt="arxiv"></a>
  <a href="https://huggingface.co/sienna223/Sim-VAE"><img src="https://img.shields.io/badge/Model-🤗-yellow" alt="model"></a>
</p>

<h4 align="center">
    <p>
        <a href=#-news>News</a> |
        <a href=#-quick-start>Quick Start</a> |
        <a href=#%EF%B8%8F-citing-us>Citation</a>
    <p>
</h4>

## 🔥 News
- **2025-07-12**: We release **Sim-VAE**, model weights can be accessed in [huggingface](https://huggingface.co/sienna223/Sim-VAE).

## Introduction
**Latent-PMRF** is new framework learning latent-generative models designed for image restoration tasks, which can achieve minimum distortion under the constrain of best perception. More specifically, consider two critical aspects interested in restoration:
- **Fidelity**: The formulation of Latent-PMRF ensures best achievable fidelity is upper bounded by VAE's reconstruction ability.
- **Visual Quality**: Since latent-space of VAE is better aligned with human perception than pixel-spacel, Latent-PMRF achieving a 5.79× speedup over PMRF in terms of FID.

Since VAE's reconstruction ability determine fidelity upper-bound of Latent-PMRF, we also propose **Sim-VAE**, which is a streamlined VAE architecture significantly outperforms existing VAEs (SD-VAE, FLUX-VAE) in both reconstruction and restoration.

<p align="center">
  <img src="assets/teaser.png" width="95%">
  <br>
  <em>Illustration of perception optimization efficiency in latent space.</em>
</p>

## 📌 TODO
- [ ] Weights of Latent-PMRF.
- [ ] Training code and guidance of Latent-PMRF.

## 🚀 Quick Start

### 🛠️ Environment Setup

#### ✅ Recommended Setup

```bash
# 1. Clone the repo
git clone git@github.com:Luciennnnnnn/Latent-PMRF.git
cd Latent-PMRF

# 2. (Optional) Create a clean Python environment
conda create -n latent-pmrf python=3.12
conda activate latent-pmrf

# 3. Install dependencies
# 3.1 Install PyTorch (choose correct CUDA version)
pip install torch==2.7.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu126

# 3.2 Install other required packages
pip install -r requirements.txt
```

#### 🌏 For users in Mainland China

```bash
# Install PyTorch from a domestic mirror
pip install torch==2.7.1 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126

# Install other dependencies from Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 🧪 Use Sim-VAE

```python
from PIL import Image

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from latent_pmrf.models.autoencoders.autoencoder_kl_sim import AutoencoderKLSim


vae = AutoencoderKLSim.from_pretrained("sienna223/Sim-VAE-c32")
vae.to(dtype=torch.float16, device="cuda")

input_image = Image.open("example_images/1.png")
input_image = to_tensor(input_image) * 2 - 1
input_image = input_image.to(device="cuda", dtype=torch.float16)

with torch.no_grad():
    output = vae(input_image.unsqueeze(0), return_dict=False)[0]

output = (output * 0.5 + 0.5).clamp(0, 1)
output_image = to_pil_image(output.squeeze(0))
output_image.save("vae_output.png")
```

## ❤️ Citing Us
If you find this repository or our work useful, please consider giving a star ⭐ and citation 🦖, which would be greatly appreciated:

```bibtex
@article{luo2025latent,
  title={Latent Posterior-Mean Rectified Flow for Higher-Fidelity Perceptual Face Restoration},
  author={Luo, Xin and Zhang, Menglin and Lan, Yunwei and Zhang, Tianyu and Li, Rui and Liu, Chang and Liu, Dong},
  journal={arXiv preprint arXiv:2507.00447},
  year={2025}
}
```
