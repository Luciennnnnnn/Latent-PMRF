import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
import time
import json
import glob
from omegaconf import OmegaConf
from tqdm import tqdm

from PIL import Image

import torch

from torchvision.transforms.functional import crop
from torchvision.transforms.functional import to_tensor, to_pil_image

from accelerate import Accelerator
from accelerate.state import AcceleratorState


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="*",
        default=['test48']
    )
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="*",
        default=['test48']
    )
    parser.add_argument(
        "--file_paths",
        type=str,
        nargs="*",
        default=['None']
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        nargs="*",
        default=[-1]
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default='full_results_diffusion_aggregation_tile512_overlap256',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=998244353
    )
    parser.add_argument(
        "--center_crop",
        action='store_true'
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default='sd1.5_vae'
    )
    parser.add_argument(
        "--test_official",
        action='store_true'
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default='fp16'
    )
    parser.add_argument(
        "--test_ema",
        action='store_true'
    )
    args = parser.parse_args()
    return args


def main(args, root_dir):
    accelerator = Accelerator(mixed_precision=args.weight_dtype if args.weight_dtype != 'fp32' else 'no')
    
    if args.weight_dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.weight_dtype == 'bf16':
        weight_dtype = torch.bfloat16
    elif args.weight_dtype == 'fp32':
        weight_dtype = torch.float32
    else:
        raise ValueError(f"Invalid weight dtype: {args.weight_dtype}")

    if args.test_official:
        if args.vae_type == 'sd1.5_vae':
            from latent_pmrf.models.autoencoders.autoencoder_kl import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="vae",
                torch_dtype=weight_dtype,
                revision="main",
                local_files_only=True,
            ).to(dtype=weight_dtype, device=accelerator.device)
        elif args.vae_type == 'sdxl_vae_fp16_fix':
            from latent_pmrf.models.autoencoders.autoencoder_kl import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                local_files_only=True,
            ).to(dtype=weight_dtype, device=accelerator.device)
        elif args.vae_type == 'flux_vae':
            from latent_pmrf.models.autoencoders.autoencoder_kl import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="vae",
                torch_dtype=weight_dtype,
                revision="main",
                local_files_only=True,
            ).to(dtype=weight_dtype, device=accelerator.device)
        vae.disable_xformers_memory_efficient_attention()
    else:
        experiment_name = args.experiment_name
        experiment_dir = os.path.join(root_dir, 'experiments', experiment_name)

        model_dir = 'vae'
        if args.test_ema:
            model_dir += '_ema'

        conf = OmegaConf.load(os.path.join(experiment_dir, f"{experiment_name}.yml"))
        from latent_pmrf.models.autoencoders.autoencoder_kl_sim import AutoencoderKLSim
        vae = AutoencoderKLSim.from_pretrained(
            experiment_dir,
            subfolder=model_dir if args.checkpoint is None else os.path.join(args.checkpoint, model_dir),
            local_files_only=True,
        ).to(device=accelerator.device)

    vae = accelerator.prepare(vae)
    vae = accelerator.unwrap_model(vae)

    output_dir = f'{root_dir}/{args.result_dir}'

    for dataset_name, dataset_dir, file_path, num_sample in zip(args.dataset_names, args.dataset_dirs, args.file_paths, args.num_samples):
        if file_path != 'None':
            with open(file_path, 'r') as f:
                image_paths = json.load(f)
            image_paths = [os.path.join(dataset_dir, image_path) for image_path in image_paths]
        else:
            image_paths = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True))

        if num_sample != -1:
            image_paths = image_paths[:num_sample]

        data_index_list = list(range(AcceleratorState().process_index, len(image_paths), AcceleratorState().num_processes))
            
        with tqdm(total=len(data_index_list), desc=f'process_index {AcceleratorState().process_index}: Processing {dataset_name} {len(data_index_list)}/{len(image_paths)}', unit='image') as pbar:
            for idx in data_index_list:
                gt_path = image_paths[idx]
                gt_pil = Image.open(gt_path).convert('RGB')
                    
                if args.center_crop:
                    gt_crop_size = args.crop_size

                    gt_crop_top = int(round((gt_pil.height - gt_crop_size) / 2.0))
                    gt_crop_left = int(round((gt_pil.width - gt_crop_size) / 2.0))

                    gt_pil = crop(gt_pil, gt_crop_top, gt_crop_left, gt_crop_size, gt_crop_size)

                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            
                gt = to_tensor(gt_pil) * 2 - 1
                gt = gt.to(device=accelerator.device, dtype=weight_dtype).unsqueeze(0)
                # with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
                pred = vae(gt, return_dict=False, generator=generator)[0]
                pred = (pred * 0.5 + 0.5).clamp(0, 1)
                pred_pil = to_pil_image(pred.squeeze(0).to(dtype=torch.float32)) # save to disk
                
                image_dir = os.path.join(output_dir, dataset_name, 'visualization')
                name, ext = os.path.splitext(os.path.relpath(gt_path, dataset_dir))
                pred_path = os.path.join(image_dir, f"{name}_pred{ext}")

                os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                pred_pil.save(pred_path)

                pbar.update(1)
           
        accelerator.wait_for_everyone()  

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)