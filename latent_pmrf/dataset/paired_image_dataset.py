import os
import json
import glob
from PIL import Image

import numpy as np

from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        self.lq_img_dir = os.path.join(self.args.data_dir, self.args.lq_img_dir)
        self.gt_img_dir = os.path.join(self.args.data_dir, self.args.gt_img_dir)

        self.lq_image_paths = glob.glob(os.path.join(self.lq_img_dir, "**/*.png"), recursive=True)
        self.lq_image_paths = sorted(self.lq_image_paths)
        self.gt_image_paths = glob.glob(os.path.join(self.gt_img_dir, "**/*.png"), recursive=True)
        self.gt_image_paths = sorted(self.gt_image_paths)

        num_samples = self.args.num_samples if 'num_samples' in self.args else len(self.lq_image_paths)
        self.lq_image_paths = self.lq_image_paths[:num_samples]
        self.gt_image_paths = self.gt_image_paths[:num_samples]

        print(f"{len(self.lq_image_paths)=}", flush=True)
    
    @property
    def features(self):
        return ['lq', 'gt']

    def __len__(self):
        return len(self.lq_image_paths)

    def __getitem__(self, idx):
        lq_image_path = self.lq_image_paths[idx]
        gt_image_path = self.gt_image_paths[idx]

        # lq_image = Image.open(lq_image_path).convert("RGB")
        # gt_image = Image.open(gt_image_path).convert("RGB")

        data = {
            "lq": {
                "path": lq_image_path,
                # "image": lq_image
            },
            "gt": {
                "path": gt_image_path,
                # "image": gt_image
            }
        }

        return data
