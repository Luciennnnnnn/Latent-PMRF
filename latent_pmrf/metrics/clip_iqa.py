r"""CLIP-IQA metric, proposed by

Exploring CLIP for Assessing the Look and Feel of Images.
Jianyi Wang Kelvin C.K. Chan Chen Change Loy.
AAAI 2023.

Ref url: https://github.com/IceClear/CLIP-IQA
Re-implmented by: Chaofeng Chen (https://github.com/chaofengc) with the following modification:
    - We assemble multiple prompts to improve the results of clipiqa model.

"""
import torch
import torch.nn as nn

from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from pyiqa.archs.arch_util import load_file_from_url
from pyiqa.archs.arch_util import load_pretrained_network

import clip
from pyiqa.archs.clip_model import load


default_model_urls = {
    'clipiqa+': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CLIP-IQA+_learned_prompts-603f3273.pth',
    'clipiqa+_rn50_512': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CLIPIQA+_RN50_512-89f5d940.pth',
    'clipiqa+_vitL14_512': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CLIPIQA+_ViTL14_512-e66488f2.pth',
}


class PromptLearner(nn.Module):
    """
    Disclaimer:
        This implementation follows exactly the official codes in: https://github.com/IceClear/CLIP-IQA. We have no idea why some tricks are implemented like this, which include
            1. Using n_ctx prefix characters "X"
            2. Appending extra "." at the end
            3. Insert the original text embedding at the middle
    """

    def __init__(self, clip_model, n_ctx=16) -> None:
        super().__init__()

        # For the following codes about prompts, we follow the official codes to get the same results
        prompt_prefix = " ".join(["X"] * n_ctx) + ' '
        init_prompts = [prompt_prefix + 'Good photo..', prompt_prefix + 'Bad photo..']
        with torch.no_grad():
            txt_token = clip.tokenize(init_prompts)
            self.tokenized_prompts = txt_token
            init_embedding = clip_model.token_embedding(txt_token)

        init_ctx = init_embedding[:, 1: 1 + n_ctx]
        self.ctx = nn.Parameter(init_ctx)

        self.n_ctx = n_ctx

        self.n_cls = len(init_prompts)
        self.name_lens = [3, 3]  # hard coded length, which does not include the extra "." at the end

        self.register_buffer("token_prefix", init_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", init_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def get_prompts_with_middel_class(self,):

        ctx = self.ctx.to(self.token_prefix)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        half_n_ctx = self.n_ctx // 2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i: i + 1, :, :]
            class_i = self.token_suffix[i: i + 1, :name_len, :]
            suffix_i = self.token_suffix[i: i + 1, name_len:, :]
            ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
            prompt = torch.cat(
                [
                    prefix_i,     # (1, 1, dim)
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_i,      # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix_i,     # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts

    def forward(self, clip_model):
        prompts = self.get_prompts_with_middel_class()
        # self.get_prompts_with_middel_class
        x = prompts + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection

        return x


class CLIPIQA(nn.Module):
    def __init__(self,
                 model_type='clipiqa',
                 backbone='RN50',
                 pretrained=True,
                 pos_embedding=False,
                 download_root: str = None
                 ) -> None:
        super().__init__()

        self.clip_model = [load(backbone, 'cpu', download_root=download_root)]  # avoid saving clip weights
        # Different from original paper, we assemble multiple prompts to improve performance
        self.prompt_pairs = clip.tokenize([
            'Good image', 'bad image',
            'Sharp image', 'blurry image',
            'sharp edges', 'blurry edges',
            'High resolution image', 'low resolution image',
            'Noise-free image', 'noisy image',
        ])

        self.model_type = model_type
        self.pos_embedding = pos_embedding
        if 'clipiqa+' in model_type:
            self.prompt_learner = PromptLearner(self.clip_model[0])

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        for p in self.clip_model[0].parameters():
            p.requires_grad = False
        
        if pretrained and 'clipiqa+' in model_type:
            if model_type == 'clipiqa+' and backbone == 'RN50':
                self.prompt_learner.ctx.data = torch.load(load_file_from_url(default_model_urls['clipiqa+']))
            elif model_type in default_model_urls.keys():
                load_pretrained_network(self, default_model_urls[model_type], True, 'params')
            else:
                raise(f'No pretrained model for {model_type}')
    
    def forward(self, x):
        # preprocess image
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        clip_model = self.clip_model[0].to(x)

        if self.model_type == 'clipiqa':
            prompts = self.prompt_pairs.to(x.device)
            logits_per_image, logits_per_text = clip_model(x, prompts, pos_embedding=self.pos_embedding)
        elif 'clipiqa+' in self.model_type:
            learned_prompt_feature = self.prompt_learner(clip_model)
            logits_per_image, logits_per_text = clip_model(
                x, None, text_features=learned_prompt_feature, pos_embedding=self.pos_embedding)

        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)

        return probs[..., 0].mean(dim=1, keepdim=True)
    

from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.img_util import imread2tensor

from pyiqa.losses.loss_util import weight_reduce_loss


class InferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(
            self,
            net,
            metric_name,
            as_loss=False,
            loss_weight=None,
            loss_reduction='mean',
            device=None,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name

        # ============ set metric properties ===========
        self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        if self.metric_mode is None:
            self.metric_mode = kwargs.pop('metric_mode')
        elif 'metric_mode' in kwargs:
            kwargs.pop('metric_mode')

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction

        self.net = net
        self.net = self.net.to(self.device)
        self.net.eval()

    def to(self, device):
        self.net.to(device)
        self.device = torch.device(device)
        return self

    def forward(self, target, ref=None, **kwargs):

        with torch.set_grad_enabled(self.as_loss):

            if 'fid' in self.metric_name:
                output = self.net(target, ref, device=self.device, **kwargs)
            else:
                if not torch.is_tensor(target):
                    target = imread2tensor(target, rgb=True)
                    target = target.unsqueeze(0)
                    if self.metric_mode == 'FR':
                        assert ref is not None, 'Please specify reference image for Full Reference metric'
                        ref = imread2tensor(ref, rgb=True)
                        ref = ref.unsqueeze(0)

                if self.metric_mode == 'FR':
                    output = self.net(target.to(self.device), ref.to(self.device), **kwargs)
                elif self.metric_mode == 'NR':
                    output = self.net(target.to(self.device), **kwargs)

        if self.as_loss:
            if isinstance(output, tuple):
                output = output[0]
            return weight_reduce_loss(output, self.loss_weight, self.loss_reduction)
        else:
            return output