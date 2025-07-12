import numpy as np
import random

import torch

from torch.utils import data as data
from torchvision import transforms


def tokenize_captions(examples, args, tokenizer, is_train=True):
    captions = []
    is_null_captions = []
    for caption in examples[args.caption_column]:
        if random.random() < args.proportion_empty_prompts:
            captions.append("")
            is_null_captions.append(True)
        elif isinstance(caption, str):
            captions.append(caption)
            is_null_captions.append(False)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
            is_null_captions.append(False)
        else:
            raise ValueError(
                f"Caption column `{args.caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids, torch.tensor(is_null_captions, dtype=torch.bool)


def text_to_img_transform(examples, args, tokenizer, is_train=True):
    transformed_examples = {}

    if is_train:
        image_transforms = transforms.Compose(
            [
                transforms.Resize(args.gt_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    else:
        image_transforms = transforms.Compose(
            [
                transforms.Resize(args.gt_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    images = [image.convert("RGB") for image in examples[args.image_column]]
    images = [image_transforms(image) for image in images]

    transformed_examples["gt"] = images
    input_ids, is_null_captions = tokenize_captions(examples, args, tokenizer, is_train)
    transformed_examples["input_ids"] = input_ids
    transformed_examples["is_null_captions"] = is_null_captions
    
    return transformed_examples