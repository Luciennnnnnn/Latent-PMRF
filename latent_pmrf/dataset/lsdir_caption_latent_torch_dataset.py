import os
import json
import glob
from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from ..data.augmentation import paired_random_crop, augment
from ..data.transform import generate_kernels

class LSDIRDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.args.only_gt = self.args.get('only_gt', False)
        
        self.image_dir = os.path.join(self.args.data_dir, self.args.img_dir)
        self.caption_dir = None
        self.latent_dir = None

        # Get list of image paths
        if self.args.file_paths is not None:
            with open(self.args.file_paths, 'r') as f:
                self.image_paths = json.load(f)
            self.image_paths = [os.path.join(self.image_dir, image_path) for image_path in self.image_paths]
        else:
            self.image_paths = glob.glob(os.path.join(self.image_dir, "**/*.png"), recursive=True)
            self.image_paths = sorted(self.image_paths)

        num_samples = self.args.num_samples if 'num_samples' in self.args else len(self.image_paths)

        if 'quality_threshold' in self.args:
            quality_score_path = os.path.join(self.args.data_dir, f"{self.args.img_dir}_quality_score.json")
            with open(quality_score_path, 'r') as f:
                quality_scores = json.load(f)
        
        if 'aesthetic_threshold' in self.args:
            aesthetic_score_path = os.path.join(self.args.data_dir, f"{self.args.img_dir}_aesthetic_score.json")
            with open(aesthetic_score_path, 'r') as f:
                aesthetic_scores = json.load(f)
            
        filtered_image_paths = []
        for image_path in self.image_paths:
            relpath, _ = os.path.splitext(os.path.relpath(image_path, self.image_dir))
            
            quality_flag = True
            aesthetic_flag = True
            # Filter by quality score
            if 'quality_threshold' in self.args:
                quality_flag = False

                # quality_score_dir = os.path.join(self.args.data_dir, self.args.quality_score_dir)

                # quality_score_path = os.path.join(quality_score_dir, f"{relpath}.json")
                # with open(quality_score_path, 'r') as f:
                #     quality_score = json.load(f)
                # if quality_score['quality'] >= self.args.quality_threshold:
                #     quality_flag = True
                if quality_scores[relpath] >= self.args.quality_threshold:
                    quality_flag = True

            if 'aesthetic_threshold' in self.args:
                aesthetic_flag = False

            #     aesthetic_score_dir = os.path.join(self.args.data_dir, self.args.aesthetic_score_dir)

            #     aesthetic_score_path = os.path.join(aesthetic_score_dir, f"{relpath}.json")
            #     with open(aesthetic_score_path, 'r') as f:
            #         aesthetic_score = json.load(f)
            #     if aesthetic_score['aesthetic'] >= self.args.aesthetic_threshold:
            #         aesthetic_flag = True
                if aesthetic_scores[relpath] >= self.args.aesthetic_threshold:
                    aesthetic_flag = True

            if quality_flag and aesthetic_flag:
                filtered_image_paths.append(image_path)
            
            if len(filtered_image_paths) >= num_samples:
                break

        self.image_paths = filtered_image_paths

        if 'face_img_dir' in self.args:
            self.face_img_dir = self.args.face_img_dir
            face_image_paths = glob.glob(os.path.join(self.face_img_dir, "**/*.png"), recursive=True)
            face_image_paths = sorted(face_image_paths)
            if 'face_num_samples' in self.args:
                face_image_paths = face_image_paths[:self.args.face_num_samples]
            
            self.image_paths = self.image_paths + face_image_paths

        print(f"{len(self.image_paths)=}", flush=True)

        if 'caption_dir' in self.args:
            self.caption_dir = os.path.join(self.args.data_dir, self.args.caption_dir)
            # Get corresponding caption and latent vector paths
            self.caption_paths = []
            for image_path in self.image_paths:
                rel_path = os.path.relpath(image_path, self.image_dir)
                self.caption_paths.append(os.path.join(self.caption_dir, rel_path.replace(".png", ".txt")))

        if 'latent_dir' in self.args:
            self.latent_dir = os.path.join(self.args.data_dir, self.args.latent_dir)
            self.latent_paths = []
            for image_path in self.image_paths:
                rel_path = os.path.relpath(image_path, self.image_dir)
                self.latent_paths.append(os.path.join(self.latent_dir, rel_path.replace(".png", ".npy")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        
        if self.latent_dir is not None:
            image = np.array(image)

            latent_path = self.latent_paths[idx]
            # latent = torch.load(latent_path).permute(1, 2, 0).numpy().astype(np.float16)
            latent = np.load(latent_path)

            image, latent = paired_random_crop(image, latent, self.args.gt_size, 8)
            image, latent = augment([image, latent], self.args.use_hflip, self.args.use_rot)

            image = to_tensor(image)
            latent = torch.tensor(np.transpose(latent, axes=(2, 0, 1)).copy(), dtype=torch.float16)
        else:
            random_crop = transforms.RandomCrop(self.args.gt_size)
            image = random_crop(image)

            image = np.array(image)
            image = augment(image, self.args.use_hflip, self.args.use_rot)
            image = to_tensor(image)

        data = {
            "gt": image,
            'image_path': image_path
        }

        # Load text caption
        if self.caption_dir is not None:
            caption_path = self.caption_paths[idx]
            if os.path.exists(caption_path):
                with open(caption_path, mode='r') as f:
                    caption = f.read()
            else:
                caption = ""
            data['caption'] = caption

        if not self.args.only_gt:
            kernel1, kernel2, sinc_kernel = generate_kernels(self.args)
            data['kernel1'] = kernel1
            data['kernel2'] = kernel2
            data['sinc_kernel'] = sinc_kernel

        if self.latent_dir is not None:
            data['latent'] = latent
            data['latent_path'] = latent_path

        return data
