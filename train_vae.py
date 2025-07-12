#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import dotenv

dotenv.load_dotenv(override=True)

import time
import gc
from copy import deepcopy
import argparse
import logging
import math
import os
import shutil
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np

import torch

import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch.distributed import ReduceOp

import transformers

import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from packaging import version
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import PretrainedConfig

import diffusers

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.image_processor import VaeImageProcessor

from latent_pmrf.data.transform import realesrgan_transform
from latent_pmrf.utils.logging_utils import TqdmToLogger

import pyiqa


if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def log_validation(vae, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    logger.info("Validation loading vae... ")
    vae = accelerator.unwrap_model(vae)

    if args.test.center_crop:
        centercrop_transform = transforms.CenterCrop(args.test.crop_size)

    image_logs = []
    vis = []

    prompt_log = ""

    from torchvision.transforms.functional import to_tensor

    logger.info("Validation images... ")
    for idx, validation_image in enumerate(args.val.validation_images):
        condition = Image.open(validation_image)
        
        if args.test.center_crop:
            condition = centercrop_transform(condition)
            
        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            
        condition = to_tensor(condition) * 2 - 1
        pred = vae(condition.to(device=accelerator.device).unsqueeze(0), return_dict=False, generator=generator)[0]

        print(f"{pred.min()=} {pred.max()=}")

        condition = condition * 0.5 + 0.5
        pred = (pred * 0.5 + 0.5).clamp(0, 1)

        vis.append(condition.cpu().squeeze(0))
        vis.append(pred.cpu().squeeze(0))

    logger.info("Save to disk... ")
    vis = torch.stack(vis, dim=0)

    from torchvision.utils import make_grid
    image_grid = make_grid(vis, nrow=vis.shape[0] // len(args.val.validation_images))
    image_grid = VaeImageProcessor.pt_to_numpy(image_grid.unsqueeze(0))
    image_grid = VaeImageProcessor.numpy_to_pil(image_grid)[0]
    image_grid.save(os.path.join(args.output_dir, f"validation_{step}.png"))

    with open(os.path.join(args.output_dir, f'validation_{step}.txt'), mode='w+') as f:
        f.write(prompt_log)

    logger.info("Finish Validation... ")

    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def _test(test_datasets, vae, args, accelerator, weight_dtype, step):
    final_test = step >= args.train.max_train_steps
        
    logger.info("Running test... ")

    logger.info("Test loading vae... ")
    vae = accelerator.unwrap_model(vae)

    from latent_pmrf.metrics.clip_iqa import CLIPIQA
    from latent_pmrf.metrics.inference_model import InferenceModel
    from torchvision.transforms.functional import to_tensor, to_pil_image

    metrics = {
        'psnr': pyiqa.create_metric('psnr', device=torch.device("cpu")),
        'ssim': pyiqa.create_metric('ssim', device=torch.device("cpu")),
        'ms_ssim': pyiqa.create_metric('ms_ssim', device=torch.device("cpu")),
        'lpips': pyiqa.create_metric('lpips', device=torch.device("cpu")),
        'dists': pyiqa.create_metric('dists', device=torch.device("cpu")),
        # 'pieapp': pyiqa.create_metric('pieapp', device=torch.device("cpu")),
        'topiq_fr': pyiqa.create_metric('topiq_fr', device=torch.device("cpu")),
        # 'niqe': pyiqa.create_metric('niqe', device=torch.device("cpu")),
        # 'brisque': pyiqa.create_metric('brisque', device=torch.device("cpu")),
        # 'clipiqa': pyiqa.create_metric('clipiqa', device=torch.device("cpu")),
        'clipiqa': InferenceModel(CLIPIQA(download_root=os.path.join(torch.hub.get_dir(), 'checkpoints')), 'clipiqa', device=torch.device("cpu")),
        'maniqa': pyiqa.create_metric('maniqa', device=torch.device("cpu")),
        'musiq': pyiqa.create_metric('musiq', device=torch.device("cpu")),
        # 'qalign': pyiqa.create_metric('qalign', device=torch.device("cpu")),
    }

    if final_test:
        result_dir = "results"
    else:
        result_dir = f"results_{step}_crop{args.test.crop_size}"

    logger.info("Test test_datasets... ")
    for dataset_name, test_dataset in test_datasets.items():
        if not final_test and dataset_name not in args.test.get('datasets_in_training', []):
            continue

        logger.info(f"Test {dataset_name=}... ")

        metric_results = {metric_name: [] for metric_name, metric in metrics.items()}

        total_time = 0
        with tqdm(total=len(test_dataset), desc='Processing', unit='image') as pbar:
            for data in test_dataset:
                gt_path = data['image']['path']
                gt_pil = Image.open(gt_path).convert('RGB')

                if args.test.center_crop:
                    crop_size = args.test.crop_size
                    crop_top = int(round((gt_pil.height - crop_size) / 2.0))
                    crop_left = int(round((gt_pil.width - crop_size) / 2.0))

                    gt_pil = crop(gt_pil, crop_top, crop_left, crop_size, crop_size)

                if args.seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    
                gt = to_tensor(gt_pil) * 2 - 1
                start = time.time()
                # with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
                pred = vae(gt.to(device=accelerator.device, dtype=weight_dtype).unsqueeze(0), return_dict=False, generator=generator)[0]
                pred = (pred * 0.5 + 0.5).clamp(0, 1)
                pred_pil = to_pil_image(pred.squeeze(0))

                total_time += time.time() - start

                # save to disk
                name, ext = os.path.splitext(os.path.basename(gt_path))

                dir_name = os.path.join(args.output_dir, result_dir, dataset_name, 'visualization')
                os.makedirs(dir_name, exist_ok=True)

                pred_pil.save(os.path.join(dir_name, f"{name}_pred{ext}"))

                # save gt to disk (temporarily)
                tmp_gt_dir_name = os.path.join(args.root_dir, "tmp_gt_dir", f"{dataset_name}{len(test_dataset)}{f'_center_crop_{args.test.crop_size}' if args.test.center_crop else ''}")
                os.makedirs(tmp_gt_dir_name, exist_ok=True)

                if not os.path.exists(os.path.join(tmp_gt_dir_name, os.path.basename(gt_path))):
                    gt_pil.save(os.path.join(tmp_gt_dir_name, os.path.basename(gt_path)))
                
                logger.info("Test calculate metrics... ")

                transform_func = transforms.Compose([
                    transforms.ToTensor(),
                ])

                pred = transform_func(pred_pil).unsqueeze(0)
                gt = transform_func(gt_pil).unsqueeze(0)

                with torch.no_grad():
                    for metric_name in metric_results.keys():
                        metric = metrics[metric_name].to(accelerator.device)
                        if metric_name in ['ssim', 'ms_ssim']:
                            metric_results[metric_name].append(metric(pred.to(torch.float64), gt.to(torch.float64)).item())
                        else:
                            metric_results[metric_name].append(metric(pred, gt).item())

                postfix = {metric_name: metric_results[metric_name][-1] for metric_name in metric_results.keys()}
                pbar.set_postfix(postfix)
                pbar.update(1)

        avg_time = total_time / len(test_dataset)
        
        logs = {}
        log_str = "\n"
        log_str += f"total time(m): {total_time / 60}\n"
        log_str += f"average time(s): {avg_time}\n"

        metric_results = {metric_name: np.mean(np.array(metric_results[metric_name])) for metric_name in metric_results.keys()}

        for metric_name, metric in metric_results.items():
            log_str += f"{metric_name}: {metric:.4f} "
            logs[f'test/{dataset_name}/{metric_name}'] = metric

        log_str += "\n------------------------------\n"

        for metric_name in metric_results.keys():
            log_str += f"{metric_name} "
        log_str += "\n"
        for metric_name, metric in metric_results.items():
            log_str += f"{metric:.4f} "
        log_str += "\n"
        
        logger.info(log_str)

        logger.info(f"Write log_str to {os.path.join(args.output_dir, result_dir, dataset_name, 'results.log')}... ")
        with open(os.path.join(args.output_dir, result_dir, dataset_name, 'results.log'), mode='a+') as f:
            f.write(log_str)
        
        accelerator.log(logs, step=step)

    logger.info("Ending test... ")

    gc.collect()
    torch.cuda.empty_cache()


def _dist_test(test_datasets, vae, args, accelerator, weight_dtype, step, num_processes, process_index):
    final_test = step >= args.train.max_train_steps
        
    logger.info("Running test... ")

    logger.info("Test loading vae... ")
    vae = accelerator.unwrap_model(vae)

    from latent_pmrf.metrics.clip_iqa import CLIPIQA
    from latent_pmrf.metrics.inference_model import InferenceModel
    from torchvision.transforms.functional import to_tensor, to_pil_image

    metrics = {
        'psnr': pyiqa.create_metric('psnr', device=torch.device("cpu")),
        'ssim': pyiqa.create_metric('ssim', device=torch.device("cpu")),
        'ms_ssim': pyiqa.create_metric('ms_ssim', device=torch.device("cpu")),
        'lpips': pyiqa.create_metric('lpips', device=torch.device("cpu")),
        'dists': pyiqa.create_metric('dists', device=torch.device("cpu")),
        # 'pieapp': pyiqa.create_metric('pieapp', device=torch.device("cpu")),
        'topiq_fr': pyiqa.create_metric('topiq_fr', device=torch.device("cpu")),
        # 'niqe': pyiqa.create_metric('niqe', device=torch.device("cpu")),
        # 'brisque': pyiqa.create_metric('brisque', device=torch.device("cpu")),
        # 'clipiqa': pyiqa.create_metric('clipiqa', device=torch.device("cpu")),
        'clipiqa': InferenceModel(CLIPIQA(download_root=os.path.join(torch.hub.get_dir(), 'checkpoints')), 'clipiqa', device=torch.device("cpu")),
        'maniqa': pyiqa.create_metric('maniqa', device=torch.device("cpu")),
        'musiq': pyiqa.create_metric('musiq', device=torch.device("cpu")),
        # 'qalign': pyiqa.create_metric('qalign', device=torch.device("cpu"))
    }

    if final_test:
        result_dir = "results"
    else:
        result_dir = f"results_{step}_crop{args.test.crop_size}"

    logger.info("Test test_datasets... ")
    for dataset_name, test_dataset in test_datasets.items():
        print(f"{dataset_name=}")
        print(f"{args.test.get('datasets_in_training', [])=}")
        if not final_test and dataset_name not in args.test.get('datasets_in_training', []):
            continue

        logger.info(f"Test {dataset_name=}... ")

        index_list = list(range(process_index, len(test_dataset), num_processes))

        metric_results = {metric_name: [] for metric_name, metric in metrics.items()}

        total_time = 0
        with tqdm(total=len(index_list), desc='Processing', unit='image') as pbar:
            for idx in index_list:
                data = test_dataset[idx]
                gt_path = data['image']['path']
                gt_pil = Image.open(gt_path).convert('RGB')

                if args.test.center_crop:
                    crop_size = args.test.crop_size
                    crop_top = int(round((gt_pil.height - crop_size) / 2.0))
                    crop_left = int(round((gt_pil.width - crop_size) / 2.0))

                    gt_pil = crop(gt_pil, crop_top, crop_left, crop_size, crop_size)

                if args.seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                gt = to_tensor(gt_pil) * 2 - 1
                start = time.time()
                # with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
                pred = vae(gt.to(device=accelerator.device, dtype=weight_dtype).unsqueeze(0), return_dict=False, generator=generator)[0]
                pred = (pred * 0.5 + 0.5).clamp(0, 1)
                pred_pil = to_pil_image(pred.squeeze(0))

                total_time += time.time() - start

                # save to disk
                name, ext = os.path.splitext(os.path.basename(gt_path))

                dir_name = os.path.join(args.output_dir, result_dir, dataset_name, 'visualization')
                os.makedirs(dir_name, exist_ok=True)

                pred_pil.save(os.path.join(dir_name, f"{name}_pred{ext}"))

                # save gt to disk (temporarily)
                tmp_gt_dir_name = os.path.join(args.root_dir, "tmp_gt_dir", f"{dataset_name}{len(test_dataset)}{f'_center_crop_{args.test.crop_size}' if args.test.center_crop else ''}")
                os.makedirs(tmp_gt_dir_name, exist_ok=True)

                if not os.path.exists(os.path.join(tmp_gt_dir_name, os.path.basename(gt_path))):
                    gt_pil.save(os.path.join(tmp_gt_dir_name, os.path.basename(gt_path)))
                
                logger.info("Test calculate metrics... ")

                transform_func = transforms.Compose([
                    transforms.ToTensor(),
                ])

                pred = transform_func(pred_pil).unsqueeze(0)
                gt = transform_func(gt_pil).unsqueeze(0)

                with torch.no_grad():
                    for metric_name in metric_results.keys():
                        metric = metrics[metric_name].to(accelerator.device)
                        if metric_name in ['ssim', 'ms_ssim']:
                            metric_results[metric_name].append(metric(pred.to(torch.float64), gt.to(torch.float64)).item())
                        else:
                            metric_results[metric_name].append(metric(pred, gt).item())

                postfix = {metric_name: metric_results[metric_name][-1] for metric_name in metric_results.keys()}
                pbar.set_postfix(postfix)
                pbar.update(1)

        avg_time = total_time / len(test_dataset)
        
        for metric_name in metric_results.keys():
            metric_results[metric_name] = torch.tensor(metric_results[metric_name], dtype=torch.float32, device=accelerator.device).sum()

        for metric_name in metric_results.keys():
            dist.all_reduce(metric_results[metric_name], op=ReduceOp.SUM)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logs = {}

            log_str = "\n"
            log_str += f"total time(m): {total_time / 60}\n"
            log_str += f"average time(s): {avg_time}\n"

            metric_results = {metric_name: metric_results[metric_name].sum().item() / len(test_dataset) for metric_name in metric_results.keys()}

            fid_metric = pyiqa.create_metric('fid', device=accelerator.device)

            dir_name = os.path.join(args.output_dir, result_dir, dataset_name, 'visualization')
            tmp_gt_dir_name = os.path.join(args.root_dir, "tmp_gt_dir", f"{dataset_name}{len(test_dataset)}{f'_center_crop_{args.test.crop_size}' if args.test.center_crop else ''}")

            metric_results['fid'] = fid_metric(dir_name, tmp_gt_dir_name).item()

            for metric_name, metric in metric_results.items():
                log_str += f"{metric_name}: {metric:.4f} "
                logs[f'test/{dataset_name}/{metric_name}'] = metric

            log_str += "\n------------------------------\n"

            for metric_name in metric_results.keys():
                log_str += f"{metric_name} "
            log_str += "\n"
            for metric_name, metric in metric_results.items():
                log_str += f"{metric:.4f} "
            log_str += "\n"
            
            logger.info(log_str)

            logger.info(f"Write log_str to {os.path.join(args.output_dir, result_dir, dataset_name, 'results.log')}... ")
            with open(os.path.join(args.output_dir, result_dir, dataset_name, 'results.log'), mode='a+') as f:
                f.write(log_str)
        
            accelerator.log(logs, step=step)

    logger.info("Ending test... ")

    gc.collect()
    torch.cuda.empty_cache()


def test(test_datasets, vae, args, accelerator, weight_dtype, step):
    num_processes = AcceleratorState().num_processes
    process_index = AcceleratorState().process_index

    if num_processes > 1:
        _dist_test(test_datasets, vae, args, accelerator, weight_dtype, step, num_processes, process_index)
    else:
        _test(test_datasets, vae, args, accelerator, weight_dtype, step)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, local_files_only: bool = False):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        local_files_only=local_files_only
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(root_path):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file.",
    )
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    output_dir = os.path.join(root_path, 'experiments', conf.name)
    
    conf.root_dir = root_path
    conf.output_dir = output_dir
    conf.config_file = args.config
    return conf


def make_train_dataset(args, tokenizer, accelerator, seed):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.get('dataset_type', 'huggingface_arrow') == 'huggingface_arrow':
        logger.info(f"***** load train dataset from disk *****")
        dataset = load_from_disk(args.train_data_dir)
    elif args.get('dataset_type', 'huggingface_arrow') == 'huggingface_dataset':
        dataset = load_dataset(**args.dataset_config)
    else:
        raise NotImplementedError(f"Do not support dataset_type {args.get('dataset_type', 'huggingface_arrow')}")

    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    logger.info(f"***** apply transform *****")
    with accelerator.main_process_first():
        if args.get('partial_dataset', None) is not None:
            assert args.get('partial_dataset', None) >= 0 and args.get('partial_dataset', None) <= 1

            logger.info(f"***** select first {int(args.get('partial_dataset', None) * len(dataset))} samples *****")

            dataset = dataset.select(range(int(args.get('partial_dataset', None) * len(dataset))))

        if args.max_train_samples is not None:
            if args.get('random_select', True):
                logger.info(f"***** random select {args.max_train_samples} samples *****")
                dataset = dataset.shuffle(seed=seed)
            else:
                logger.info(f"***** select first {args.max_train_samples} samples *****")

            dataset = dataset.select(range(args.max_train_samples))
        # Set the training transforms
        transform = partial(realesrgan_transform, tokenizer=tokenizer, args=args)
        train_dataset = dataset.with_transform(transform)

    return train_dataset

def create_test_dataset(args):
    logger.info(f"***** create test dataset *****")
    
    test_datasets = {}
    for dataset_config in args.test.datasets:
        config = deepcopy(dataset_config)
        dataset_name = config.pop('dataset_name')
        dataset_type = config.pop('dataset_type', 'arrow')
        if dataset_type == 'ImageDataset':
            from latent_pmrf.dataset.image_dataset import ImageDataset
            dataset = ImageDataset(config)
        else:
            dataset = load_dataset(**config, trust_remote_code=True)

        test_datasets[dataset_name] = dataset
    
    return test_datasets


@torch.no_grad()
def ema_step(ema_model, model, decay=0.9999):
    for ema_param, param in zip(list(ema_model.parameters()), list(model.parameters())):
        if param.requires_grad:
            ema_param.sub_((1 - decay) * (ema_param - param))
        else:
            ema_param.copy_(param)


def predict(transformer, noisy_latents, timesteps, control_cond):
    # Predict the noise residual
    model_pred = transformer(noisy_latents, sigma=timesteps, cond=control_cond)

    return model_pred


def setup_logging(args: OmegaConf, accelerator: Accelerator) -> None:
    """
    Set up logging configuration for training.
    
    Args:
        accelerator: Accelerator instance
        args: Configuration object
        logging_dir: Directory for log files
    """

    logging_dir = Path(args.output_dir, "logs")
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config_file, args.output_dir)
        
        # Create logging directory and file handler
        os.makedirs(logging_dir, exist_ok=True)
        log_file = Path(logging_dir, f'{time.strftime("%Y%m%d-%H%M%S")}.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.logger.addHandler(file_handler)

    # Configure basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set verbosity for different processes
    log_level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    transformers.utils.logging.set_verbosity(log_level)
    diffusers.utils.logging.set_verbosity(log_level)
    
def main(args):
    logging_dir = Path(args.output_dir, 'logs')

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        mixed_precision=args.train.mixed_precision,
        log_with=args.logger.log_with,
        project_config=accelerator_project_config,
    )

    setup_logging(args, accelerator)
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=args.get('device_specific_seed', False))

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    if 'pretrained_vae_name_or_path' in args.model:
        from latent_pmrf.models.res_autoencoders.autoencoder_kl_sim import AutoencoderKLSim
        vae = AutoencoderKLSim.from_pretrained(
            args.model.pretrained_vae_name_or_path,
            in_channels=args.model.arch_opt.in_channels,
            out_channels=args.model.arch_opt.out_channels,
            down_block_types=tuple(args.model.arch_opt.down_block_types),
            up_block_types=tuple(args.model.arch_opt.up_block_types),
            block_out_channels=tuple(args.model.arch_opt.block_out_channels),
            layers_per_block=args.model.arch_opt.layers_per_block,
            layers_mid_block=args.model.arch_opt.layers_mid_block,
            latent_channels=args.model.arch_opt.latent_channels,
            norm_type=args.model.arch_opt.get('norm_type', 'layer_norm'),
            force_upcast=args.model.arch_opt.force_upcast,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
    else:
        from latent_pmrf.models.res_autoencoders.autoencoder_kl_sim import AutoencoderKLSim
        vae = AutoencoderKLSim(
            in_channels=args.model.arch_opt.in_channels,
            out_channels=args.model.arch_opt.out_channels,
            down_block_types=tuple(args.model.arch_opt.down_block_types),
            up_block_types=tuple(args.model.arch_opt.up_block_types),
            block_out_channels=tuple(args.model.arch_opt.block_out_channels),
            mid_block_type=args.model.arch_opt.mid_block_type,
            mid_block_out_channel=args.model.arch_opt.mid_block_out_channel,
            layers_per_block=args.model.arch_opt.layers_per_block,
            layers_mid_block=args.model.arch_opt.layers_mid_block,
            latent_channels=args.model.arch_opt.latent_channels,
            norm_type=args.model.arch_opt.get('norm_type', 'layer_norm'),
            force_upcast=args.model.arch_opt.force_upcast,
        )
    
    if args.train.get('torch_compile', False):
        logger.info("***** torch compile *****")
        vae = torch.compile(vae, mode="default", fullgraph=args.train.get('torch_compile_fullgraph', True))

    ema_decay = args.train.get('ema_decay', 0)

    if ema_decay != 0:
        vae_ema = deepcopy(vae)
        vae_ema = EMAModel(vae_ema.parameters(), decay=ema_decay, model_cls=type(unwrap_model(vae)), model_config=vae_ema.config)
        
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.train.get('benchmark_cudnn', False):
        torch.backends.cudnn.benchmark = True

    if args.train.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if ema_decay != 0:
                    vae_ema.save_pretrained(os.path.join(output_dir, "vae_ema"))
                
                for i, model in enumerate(models):
                    sub_dir = "vae"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if ema_decay != 0:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vae_ema"), type(unwrap_model(vae)))
                vae_ema.load_state_dict(load_model.state_dict())
                vae_ema.to(accelerator.device)
                del load_model
            
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = type(unwrap_model(vae)).from_pretrained(input_dir, subfolder="vae")

                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.train.scale_lr:
        args.train.learning_rate = (
            args.train.learning_rate * args.train.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    vae_params = list(filter(lambda p: p.requires_grad, vae.parameters()))

    total_params = sum(p.numel() for p in vae.parameters())
    total_trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    logger.info(f"Number of parameters (M): {total_params / 1000 / 1000}")
    logger.info(f"Number of trainable parameters (M): {total_trainable_params / 1000 / 1000}")

    logger.info(vae)

    optimizer = optimizer_class(
        vae_params,
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )

    logger.info("***** Prepare dataset *****")
    if 'FFHQ' in args.data.dataset_name:
        from latent_pmrf.dataset.ffhq_torch_dataset import FFHQDataset
        train_dataset = FFHQDataset(args=args.data)
    else:
        from latent_pmrf.dataset.lsdir_caption_latent_torch_dataset import LSDIRDataset
        train_dataset = LSDIRDataset(args=args.data)

    test_datasets = create_test_dataset(args)

    if args.seed is not None and args.get('workder_specific_seed', False):
        from latent_pmrf.utils.reproducibility import worker_init_fn
        worker_init_fn = partial(worker_init_fn,
                                num_processes=AcceleratorState().num_processes,
                                num_workers=args.train.dataloader_num_workers,
                                process_index=AcceleratorState().process_index,
                                seed=args.seed,
                                same_seed_per_epoch=args.get('same_seed_per_epoch', False))
    else:
        worker_init_fn = None

    logger.info("***** Prepare dataLoader *****")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train.batch_size,
        num_workers=args.train.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if 'max_train_steps' not in args.train:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.train.lr_scheduler == 'timm_cosine':
        from latent_pmrf.optim.scheduler.cosine_lr import CosineLRScheduler

        lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                         t_initial=args.train.t_initial,
                                         lr_min=args.train.lr_min,
                                         cycle_decay=args.train.cycle_decay,
                                         warmup_t=args.train.warmup_t,
                                         warmup_lr_init=args.train.warmup_lr_init,
                                         warmup_prefix=args.train.warmup_prefix,
                                         t_in_epochs=args.train.t_in_epochs)
    else:
        lr_scheduler = get_scheduler(
            args.train.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.train.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.train.max_train_steps * accelerator.num_processes,
            num_cycles=args.train.lr_num_cycles,
            power=args.train.lr_power,
        )

    logger.info("***** Prepare everything with our accelerator *****")
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )

    if ema_decay != 0:
        vae_ema.to(accelerator.device)

    lpips_func = pyiqa.create_metric('lpips', net='vgg', as_loss=True, device=accelerator.device)

    args.train.perceptual_loss_weight = args.train.get('perceptual_loss_weight', 0)
    if args.train.perceptual_loss_weight > 0:
        from latent_pmrf.losses.perceptual_loss import PerceptualLoss
        perceptual_loss_func = PerceptualLoss(
            layer_weights=args.train.perceptual_loss_layer_weights,
            vgg_type='vgg19',
            use_input_norm=True,
            range_norm=True,
            perceptual_weight=args.train.perceptual_loss_weight,
            style_weight=0.,
            criterion='l1',
        ).to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train.num_train_epochs = math.ceil(args.train.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(args)
        accelerator.init_trackers(args.logger.get('project_name', 'vae'), config=tracker_config, init_kwargs={"wandb": {"name": args.name}})

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.get('resume_from_checkpoint_external', False):
            path = args.resume_from_checkpoint
        elif args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if args.get('resume_from_checkpoint_external', False):
                accelerator.load_state(path)
            else:
                accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        file=TqdmToLogger(logger, level=logging.INFO)
    )

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                logger.info(f"***** Wandb log dir: {tracker.run.dir} *****")

    # if global_step == 0 and edm_sde is False:
    if global_step == 0 and not args.get('skip_test', False):
        test(test_datasets, vae, args, accelerator, weight_dtype, 0)

    for epoch in range(first_epoch, args.train.num_train_epochs):
        if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
            break
        for step, batch in enumerate(train_dataloader):
            gt = batch['gt']
            gt = VaeImageProcessor.normalize(gt)
            
            with accelerator.accumulate(vae):
                posterior = accelerator.unwrap_model(vae).encode(gt, return_dict=False)[0]
                z = posterior.sample()
                pred = accelerator.unwrap_model(vae).decode(z, return_dict=False)[0]
                
                if args.train.rec_loss_type == 'l1':
                    pixel_loss = F.l1_loss(pred.float(), gt.float(), reduction="none").mean()
                elif args.train.rec_loss_type == 'mse':
                    pixel_loss = F.mse_loss(pred.float(), gt.float(), reduction="none").mean()
                kl_loss = args.train.kl_loss_weight * posterior.kl().mean()

                loss = pixel_loss + kl_loss

                if args.train.lpips_loss_weight > 0:
                    lpips_loss = args.train.lpips_loss_weight * lpips_func(pred, gt).mean()
                    loss += lpips_loss

                if args.train.perceptual_loss_weight > 0:
                    perceptual_loss, _ = perceptual_loss_func(pred, gt)
                    loss += perceptual_loss.mean()

                accelerator.backward(loss)

                avg_loss = accelerator.gather(loss.repeat(args.train.batch_size)).mean()
                avg_pixel_loss = accelerator.gather(pixel_loss.repeat(args.train.batch_size)).mean()
                avg_kl_loss = accelerator.gather(kl_loss.repeat(args.train.batch_size)).mean()
                if args.train.lpips_loss_weight > 0:
                    avg_lpips_loss = accelerator.gather(lpips_loss.repeat(args.train.batch_size)).mean()
                if args.train.perceptual_loss_weight > 0:
                    avg_perceptual_loss = accelerator.gather(perceptual_loss.repeat(args.train.batch_size)).mean()
                
                if accelerator.sync_gradients:
                    params_to_clip = list(vae.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.train.max_grad_norm)
                optimizer.step()
                if 'timm' in args.train.lr_scheduler:
                    lr_scheduler.step(global_step)
                else:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.train.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if ema_decay != 0:
                    vae_ema.step(vae.parameters())
                    
                global_step += 1

                logs = {"loss": avg_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                logs['pixel_loss'] = avg_pixel_loss.detach().item()
                if args.train.lpips_loss_weight > 0:
                    logs['lpips_loss'] = avg_lpips_loss.detach().item()
                if args.train.perceptual_loss_weight > 0:
                    logs['perceptual_loss'] = avg_perceptual_loss.detach().item()
                
                logs['kl_loss'] = avg_kl_loss.detach().item()
                
                if accelerator.is_main_process:
                    if global_step % args.logger.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.logger.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.logger.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    
                    if 'train_visualization_steps' in args.val and (global_step - 1) % args.val.train_visualization_steps == 0:
                        num_imgs = min(3, gt.shape[0])
                        with torch.no_grad():
                            image = torch.cat([gt[:num_imgs], pred[:num_imgs]], dim=0)
                            image = VaeImageProcessor.denormalize(image)
                                
                            from torchvision.utils import make_grid
                            for i in range(num_imgs):
                                image_grid = make_grid(image[i::num_imgs], nrow=image[i::num_imgs].shape[0])
                                image_grid = VaeImageProcessor.pt_to_numpy(image_grid.unsqueeze(0))
                                image_grid = VaeImageProcessor.numpy_to_pil(image_grid)[0]
                                image_grid.save(os.path.join(args.output_dir, f"train_visualization_{global_step}_{i}.png"))

                if not args.get('skip_test', False) and global_step % args.test.get('test_steps', args.train.max_train_steps) == 0 and global_step < args.train.max_train_steps:
                    if ema_decay != 0:
                        # Store the Controlnet parameters temporarily and load the EMA parameters to perform inference.
                        vae_ema.store(vae.parameters())
                        vae_ema.copy_to(vae.parameters())
                        logger.info(f"Test loading ema vae... {ema_decay=}")

                    # test_datasets = create_test_dataset(args)
                    test(
                        test_datasets,
                        vae,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )
                    
                    if ema_decay != 0:
                        # Switch back to the original UNet parameters.
                        vae_ema.restore(vae.parameters())

                if accelerator.is_main_process:
                    if args.val.validation_steps != -1 and (global_step - 1) % args.val.validation_steps == 0:
                        if ema_decay != 0:
                            # Store the Controlnet parameters temporarily and load the EMA parameters to perform inference.
                            vae_ema.store(vae.parameters())
                            vae_ema.copy_to(vae.parameters())

                        image_logs = log_validation(
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step
                        )
                        
                        if ema_decay != 0:
                            # Switch back to the original UNet parameters.
                            vae_ema.restore(vae.parameters())
                                    
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

                accelerator.log(logs, step=global_step)
            if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
                break

            # data_start = time.time()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) > 0 and int(checkpoints[-1].split("-")[1]) < global_step:
            if args.logger.checkpoints_total_limit is not None:

                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= args.logger.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(os.path.join(args.output_dir, 'vae'))

        if ema_decay != 0:
            vae_ema.save_pretrained(os.path.join(args.output_dir, 'vae_ema'))
    
    if not args.get('skip_test', False):
        if ema_decay != 0:
            # Store the Controlnet parameters temporarily and load the EMA parameters to perform inference.
            vae_ema.store(vae.parameters())
            vae_ema.copy_to(vae.parameters())
        test(
            test_datasets,
            vae,
            args,
            accelerator,
            weight_dtype,
            global_step,
        )
        if ema_decay != 0:
            # Switch back to the original UNet parameters.
            vae_ema.restore(vae.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args(root_path)
    main(args)