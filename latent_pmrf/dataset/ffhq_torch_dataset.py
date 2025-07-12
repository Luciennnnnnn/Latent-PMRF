import os
import json
import glob
from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision.transforms.functional import to_tensor

from ..data.augmentation import paired_random_crop, augment
from .pmrf.create_degradation import create_degradation

class FFHQDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.args.file_paths = self.args.get('file_paths', None)
        self.args.only_gt = self.args.get('only_gt', False)
        
        self.image_dir = os.path.join(self.args.data_dir, self.args.img_dir)

        # Get list of image paths
        if self.args.file_paths is not None:
            with open(self.args.file_paths, 'r') as f:
                self.image_paths = json.load(f)
            self.image_paths = [os.path.join(self.image_dir, image_path) for image_path in self.image_paths]
        else:
            self.image_paths = glob.glob(os.path.join(self.image_dir, "**/*.png"), recursive=True)
            self.image_paths = sorted(self.image_paths)

        num_samples = self.args.num_samples if 'num_samples' in self.args else len(self.image_paths)

        filtered_image_paths = self.image_paths[:num_samples]

        self.image_paths = filtered_image_paths

        print(f"{len(self.image_paths)=}", flush=True)

        self.latent_dir = None
        if 'latent_dir' in self.args:
            self.latent_dir = os.path.join(self.args.data_dir, self.args.latent_dir)
            self.latent_paths = []
            for image_path in self.image_paths:
                rel_path = os.path.relpath(image_path, self.image_dir)
                self.latent_paths.append(os.path.join(self.latent_dir, rel_path.replace(".png", ".npy")))
        
        if not self.args.only_gt:
            self.degradation = create_degradation(self.args.degradation_model)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        if self.latent_dir is not None:
            gt = np.array(image) / 255.0

            latent_path = self.latent_paths[idx]
            # latent = torch.load(latent_path).permute(1, 2, 0).numpy().astype(np.float16)
            latent = np.load(latent_path)

            gt, latent = paired_random_crop(gt, latent, self.args.gt_size, self.args.latent_scale)
            # image, latent = augment([image, latent], self.args.use_hflip, self.args.use_rot) # disable augmentation since latent is not augmentation friendly

            latent = torch.tensor(np.transpose(latent, axes=(2, 0, 1)).copy(), dtype=torch.float16)
        else:
            gt = np.array(image) / 255.0
            gt = augment(gt, self.args.use_hflip, self.args.use_rot)

        if not self.args.only_gt:
            # import torch
            # gt = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
            lq, maybe_gt = self.degradation(gt)
            if maybe_gt is not None:
                gt = maybe_gt

            data = {
                "gt": gt,
                "lq": lq,
                'image_path': image_path
            }
            if self.latent_dir is not None:
                data['latent'] = latent
        else:
            data = {
                "gt": gt,
                'image_path': image_path
            }
            if self.latent_dir is not None:
                data['latent'] = latent

        return data
