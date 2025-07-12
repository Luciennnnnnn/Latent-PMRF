import dotenv

dotenv.load_dotenv(override=True)

from PIL import Image

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from latent_pmrf.models.autoencoders.autoencoder_kl_sim import AutoencoderKLSim


vae = AutoencoderKLSim.from_pretrained("sienna223/Sim-VAE")
vae.to(dtype=torch.float16, device="cuda")

input_image = Image.open("example_images/1.png")
input_image = to_tensor(input_image) * 2 - 1
input_image = input_image.to(device="cuda", dtype=torch.float16)

with torch.no_grad():
    output = vae(input_image.unsqueeze(0), return_dict=False)[0]

output = (output * 0.5 + 0.5).clamp(0, 1)
output_image = to_pil_image(output.squeeze(0))
output_image.save("vae_output.png")