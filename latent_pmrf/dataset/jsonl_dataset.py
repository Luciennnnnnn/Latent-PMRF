from typing import Optional

import math
import json
from PIL import Image

import torch

class JsonlDataset(torch.utils.data.Dataset):
    possible_read_keys = [
        "output_image",
        "input_image",
        "input_images",
        "instruction",
        "input_description",
        "input_descriptions",
        "output_description",
        "has_human",
    ]

    def __init__(
        self,
        data_files,
        num_images: Optional[int] = None,
        largest_side: Optional[int] = None,
        largest_pixels: Optional[int] = None,
        read_keys=["output_image", "instruction"],
    ):
        self.data_files = data_files
        self.largest_side = largest_side
        self.largest_pixels = largest_pixels
        self.read_keys = read_keys
        self.datas = []
        
        for data_file in data_files:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    json_data = json.loads(line.strip())
                    data = {'source_file': data_file, 'json_data': json_data}

                    for key in self.possible_read_keys:
                        if key in json_data:
                            data[key] = json_data[key]

                    self.datas.append(data)
                    if num_images is not None and len(self.datas) >= num_images:
                        break
            # print(f"{data_file=} {len(self.datas)=}")
            if num_images is not None and len(self.datas) >= num_images:
                break

    def __len__(self):
        return len(self.datas)

    def resize_image(self, pil_image):
        width, height = pil_image.size
        if self.largest_side is not None and max(width, height) > self.largest_side:
            scale = self.largest_side / max(width, height)
            pil_image = pil_image.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
        
        width, height = pil_image.size
        if self.largest_pixels is not None and width * height > self.largest_pixels:
            scale = math.sqrt(self.largest_pixels / (width * height))
            pil_image = pil_image.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
        return pil_image

    def __getitem__(self, index):
        data = {}
        data['source_file'] = self.datas[index]['source_file']
        data['json_data'] = self.datas[index]['json_data']
        if 'output_image' in self.read_keys:
            output_image_path = self.datas[index]['output_image']
            pil_image = Image.open(output_image_path).convert('RGB')
            pil_image = self.resize_image(pil_image)
            
            data['output_image'] = pil_image
            data['output_image_path'] = output_image_path

        assert not ('input_image' in self.datas[index] and 'input_images' in self.datas[index])

        if 'input_image' in self.read_keys:
            input_image_path = self.datas[index]['input_image']
            if isinstance(input_image_path, list):
                pil_images = [Image.open(input_image_path).convert('RGB') for input_image_path in input_image_path]
                pil_images = [self.resize_image(pil_image) for pil_image in pil_images]

                data['input_images'] = pil_images
            else:
                pil_image = Image.open(input_image_path).convert('RGB')
                pil_image = self.resize_image(pil_image)
                data['input_images'] = pil_image

            data['input_image_paths'] = input_image_path
            
        if 'input_images' in self.read_keys:
            input_image_paths = self.datas[index]['input_images']
            pil_images = [Image.open(input_image_path).convert('RGB') for input_image_path in input_image_paths]
            pil_images = [self.resize_image(pil_image) for pil_image in pil_images]

            data['input_images'] = pil_images
            data['input_image_paths'] = input_image_paths

        if 'instruction' in self.read_keys:
            data['instruction'] = self.datas[index]['instruction']

        if 'input_description' in self.read_keys:
            data['input_description'] = self.datas[index]['input_description']

        if 'output_description' in self.read_keys:
            data['output_description'] = self.datas[index]['output_description']
        
        if 'has_human' in self.read_keys:
            data['has_human'] = self.datas[index]['has_human']
            
        return data