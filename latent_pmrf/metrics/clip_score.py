from typing import Any

import torch
from torch import nn

from transformers import CLIPProcessor, CLIPModel    

class CLIPScore(nn.Module):
    def __init__(self, clip_model_name_or_path="openai/clip-vit-large-patch14") -> None:
        super().__init__()

        self.clip_model = CLIPModel.from_pretrained(clip_model_name_or_path)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name_or_path)
        
    @torch.no_grad()
    def forward(self, caption, image) -> Any:
        inputs = self.clip_processor(text=caption, images=image, return_tensors="pt")
        input_ids = inputs.input_ids.to(device=self.clip_model.device)

        shape_max_length = input_ids.shape[-1]

        text_embeds_list = []
        for i in range(0, shape_max_length, self.clip_processor.tokenizer.model_max_length):
            text_embeds = self.clip_model.get_text_features(input_ids[:, i: i + self.clip_processor.tokenizer.model_max_length])
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds_list.append(text_embeds)

        image = inputs.pixel_values.to(device=self.clip_model.device)
        image_embeds = self.clip_model.get_image_features(image)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = []
        for text_embeds in text_embeds_list:
            # cosine similarity as logits
            logits_per_text.append(torch.matmul(text_embeds, image_embeds.t()))

        logits_per_text = torch.stack(logits_per_text, dim=-1)

        logits_per_text_max, _ = logits_per_text.max(dim=-1)
        logits_per_text_max = logits_per_text_max.mean()

        logits_per_text_mean = logits_per_text.mean(dim=-1).mean()

        return logits_per_text_max, logits_per_text_mean