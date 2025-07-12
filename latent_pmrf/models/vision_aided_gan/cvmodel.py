import importlib
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import timm
import clip

from .DiffAugment_pytorch import DiffAugment


class CLIP(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model, _ = clip.load("ViT-B/32", jit=False, device='cpu')
        self.model = self.model.visual
        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
 
    def forward_custom(self, x):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x1 = []
        feat_points = [0, 4, 8, len(self.model.transformer.resblocks)]
        for i in range(len(feat_points)-1):
            x = self.model.transformer.resblocks[feat_points[i]:feat_points[i+1]](x)
            x1.append(x.permute(1, 0, 2))

        x = self.model.ln_post(x1[-1][:, 0, :])
        if self.model.proj is not None:
            x = x @ self.model.proj
        x1[-1] = x
        return x1

    def __call__(self, x):
        x = F.interpolate(x*0.5+0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)
            
        if 'conv_multi_level' in self.cv_type:
            x = self.forward_custom(x.type(self.model.conv1.weight.dtype))
            x[0] = x[0][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7).float()
            x[1] = x[1][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7).float()
            x[2] = x[2].float()
        else:
            x = self.model(x.type(self.model.conv1.weight.dtype)).float()
            
        return x
    
class DINO(torch.nn.Module):

    def __init__(self, cv_type='adv', input_resolution: int = 224):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.model.eval()
        self.model.requires_grad = False
        self.input_resolution = input_resolution
        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])
       
    def __call__(self, x):
        x = F.interpolate(x*0.5+0.5, size=(self.input_resolution, self.input_resolution), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)
        
        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8)
            x = [x[i] for i in [0, 4, -1]]
            x[0] = x[0][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            x[1] = x[1][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            x[2] = x[2][:, 0, :]
        else:
            x = self.model(x)
            
        return x

class DINOv2(torch.nn.Module):

    def __init__(self, cv_type='adv', input_resolution: int = 224):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        self.model.requires_grad = False
        self.input_resolution = input_resolution
        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])
       
    def __call__(self, x):
        x = F.interpolate(x*0.5+0.5, size=(self.input_resolution, self.input_resolution), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)
        
        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8, reshape=True, return_class_token=True)
            x = [x[i] for i in [0, 4, -1]]
            x[0] = x[0][0]
            x[1] = x[1][0]
            x[2] = x[2][1]
        else:
            x = self.model(x)
        return x


class DINOv2Reg(torch.nn.Module):

    def __init__(self, cv_type='adv', input_resolution: int = 224):
        super().__init__()

        self.cv_type = cv_type
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        print(f"{self.model=}")
        self.model.eval()
        self.model.requires_grad = False
        self.input_resolution = input_resolution
        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])

        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [self.input_resolution // 14, self.input_resolution // 14],
        )
        
    def __call__(self, x):
        x = F.interpolate(x * 0.5 + 0.5, size=(self.input_resolution, self.input_resolution), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)
        
        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8, reshape=True, return_class_token=True)
            x = [x[i] for i in [0, 4, -1]]
            x[0] = x[0][0]
            x[1] = x[1][0]
            x[2] = x[2][1]
        else:
            x = self.model(x)
            
        return x
    
class CVBackbone(torch.nn.Module):

    def __init__(self, cv_type, output_type, diffaug=False, input_resolution: int = 224):
        super().__init__(
        )
        cv_type = cv_type.split('+')
        output_type = output_type.split('+')
        self.class_name_dict = {
                'seg_ade': 'vision_aided_loss.swintaskspecific.Swin',
                'det_coco': 'vision_aided_loss.swintaskspecific.Swin',
                'clip': 'vision_aided_loss.cvmodel.CLIP',
                'dino': 'vision_aided_loss.cvmodel.DINO',
                'vgg': 'vision_aided_loss.cvmodel.Vgg',
                'swin': 'vision_aided_loss.cvmodel.Swin',
                'face_seg': 'vision_aided_loss.face_parsing.Parsing',
                'face_normals': 'vision_aided_loss.face_normals.Normals'
            }

        self.cv_type = cv_type
        self.policy = ''
        if diffaug:
            self.policy = 'color,translation,cutout'
            
        self.models = []
        for cv_type_, output_type_ in zip(cv_type, output_type):
            # modellib = importlib.import_module('.'.join(self.class_name_dict[cv_type_].split('.')[:-1]))
            # model = None
            # target_model_name = self.class_name_dict[cv_type_].split('.')[-1]
            # for name, cls in modellib.__dict__.items():
            #     if name.lower() == target_model_name.lower():
            #         model = cls
            print(f"{cv_type_=}")
            if cv_type_ == 'dino':
                model = DINO
            elif cv_type_ == 'dinov2':
                model = DINOv2
            elif cv_type_ == 'dinov2_reg':
                model = DINOv2Reg
            elif cv_type_ == 'clip':
                model = CLIP
                    
            cv_type_ = cv_type_ + '_' + output_type_
            model = model(cv_type=cv_type_, input_resolution=input_resolution).requires_grad_(False)
            self.models.append(model)
    
    def to(self, *args, **kwargs):
        for model in self.models:
            model = model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def __call__(self, images):
        image_features = []
        for i, each in enumerate(self.models):
            image_features.append(each(DiffAugment(images, policy=self.policy)))
        return image_features