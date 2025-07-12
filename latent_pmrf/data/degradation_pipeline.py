import random
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils import data as data

from .degradation import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from .augmentation import paired_random_crop
from ..utils.diffjpeg import DiffJPEG
from ..utils.img_process_util import USMSharp, filter2D


class RealESRGANDegradationModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp()  # do usm sharpening
        self.queue_size = args.queue_size
        self.queue_ptr = 0

        self.args = args

    @torch.no_grad()
    def _dequeue_and_enqueue(self, data):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        if not hasattr(self, 'queue_lr'):
            for k, v in data.items():
                if not hasattr(self, f'queue_{k}'):
                    assert self.queue_size % v.size(0) == 0, f'queue size {self.queue_size} should be divisible by batch size {v.size(0)}'
                    setattr(self, f'queue_{k}', torch.zeros(self.queue_size, *v.size()[1:], dtype=v.dtype, device=v.device))
            
        if self.queue_ptr == self.queue_size:
            # shuffle
            idx = torch.randperm(self.queue_size)
            results = {}
            for k, v in data.items():
                setattr(self, f'queue_{k}', getattr(self, f'queue_{k}')[idx])

                # get first b samples
                results[k] = getattr(self, f'queue_{k}')[0:v.size(0)].clone()
                # update the queue
                getattr(self, f'queue_{k}')[0:v.size(0)] = v.clone()
            return results
        else:
            for k, v in data.items():
                getattr(self, f'queue_{k}')[self.queue_ptr:self.queue_ptr + v.size(0)] = v.clone()
            self.queue_ptr = self.queue_ptr + v.size(0)
            return data

        # # initialize
        # b, c, h, w = lq.size()
        # if not hasattr(self, 'queue_lr'):
        #     assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
        #     self.queue_lr = torch.zeros(self.queue_size, c, h, w, dtype=lq.dtype, device=lq.device)
        #     _, c, h, w = gt.size()
        #     self.queue_gt = torch.zeros(self.queue_size, c, h, w, dtype=gt.dtype, device=gt.device)
        #     if input_ids is not None:
        #         _, n = input_ids.size()
        #         self.queue_input_ids = torch.zeros(self.queue_size, n, dtype=input_ids.dtype, device=input_ids.device)
        #     self.queue_ptr = 0
        # if self.queue_ptr == self.queue_size:  # the pool is full
        #     # do dequeue and enqueue
        #     # shuffle
        #     idx = torch.randperm(self.queue_size)
        #     self.queue_lr = self.queue_lr[idx]
        #     self.queue_gt = self.queue_gt[idx]
        #     # get first b samples
        #     lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
        #     gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
        #     # update the queue
        #     self.queue_lr[0:b, :, :, :] = lq.clone()
        #     self.queue_gt[0:b, :, :, :] = gt.clone()
        #     if input_ids is not None:
        #         self.queue_input_ids = self.queue_input_ids[idx]
        #         # get first b samples
        #         input_ids_dequeue = self.queue_input_ids[0:b, :].clone()
        #         # update the queue
        #         self.queue_input_ids[0:b, :] = input_ids.clone()

        #         return lq_dequeue, gt_dequeue, input_ids_dequeue

        #     return lq_dequeue, gt_dequeue
        # else:
        #     # only do enqueue
        #     self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = lq.clone()
        #     self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = gt.clone()
        #     if input_ids is not None:
        #         self.queue_input_ids[self.queue_ptr:self.queue_ptr + b, :] = input_ids.clone()
        #     self.queue_ptr = self.queue_ptr + b

        #     if input_ids is not None:
        #         return lq, gt, input_ids
            
        #     return lq, gt

    @torch.no_grad()
    def __call__(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # training data synthesis
        gt = data['gt']
        if self.args.use_usm:
            gt = self.usm_sharpener(gt)

        kernel1 = data['kernel1']
        kernel2 = data['kernel2']
        sinc_kernel = data['sinc_kernel']

        ori_h, ori_w = gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.args.get('gating_p', 1):
            out = filter2D(gt, kernel1)
        else:
            out = gt

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.args.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.args.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.args.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < self.args.get('gating_p', 1):
            gray_noise_prob = self.args.gray_noise_prob
            if np.random.uniform() < self.args.gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.args.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.args.poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
        # JPEG compression
        if np.random.uniform() < self.args.get('gating_p', 1):
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.args.jpeg_range)
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifactss
            out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.args.get('gating_p', 1):
            if np.random.uniform() < self.args.second_blur_prob:
                out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.args.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.args.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.args.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.args.scale * scale), int(ori_w / self.args.scale * scale)), mode=mode)
        # add noise
        if np.random.uniform() < self.args.get('gating_p', 1):
            gray_noise_prob = self.args.gray_noise_prob2
            if np.random.uniform() < self.args.gaussian_noise_prob2:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.args.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.args.poisson_scale_range2,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.args.scale, ori_w // self.args.scale), mode=mode)
            if np.random.uniform() < self.args.get('gating_p', 1):
                out = filter2D(out, sinc_kernel)
            out = torch.clamp(out, 0, 1)
            # JPEG compression
            if np.random.uniform() < self.args.get('gating_p', 1):
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.args.jpeg_range2)
                out = self.jpeger(out, quality=jpeg_p)
        else:
            out = torch.clamp(out, 0, 1)
            # JPEG compression
            if np.random.uniform() < self.args.get('gating_p', 1):
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.args.jpeg_range2)
                out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.args.scale, ori_w // self.args.scale), mode=mode)
            if np.random.uniform() < self.args.get('gating_p', 1):
                out = filter2D(out, sinc_kernel)

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        if self.args.get('patch_aware', False):
            global_lq = lq

            b, c, h_lq, w_lq = lq.shape
            lq_size = self.args.gt_size // self.args.scale

            top = torch.randint(0, h_lq - lq_size + 1, (b,), device=lq.device)
            left = torch.randint(0, w_lq - lq_size + 1, (b,), device=lq.device)

            # Create a meshgrid for the patches
            i = torch.arange(lq_size, device=lq.device).view(1, 1, -1, 1)
            j = torch.arange(lq_size, device=lq.device).view(1, 1, 1, -1)

            # Offset the indices by the top-left corner positions
            i = i + top.view(-1, 1, 1, 1)
            j = j + left.view(-1, 1, 1, 1)

            batch_indices = torch.arange(b, device=lq.device).view(-1, 1, 1, 1)
            channel_indices = torch.arange(c, device=lq.device).view(1, -1, 1, 1)

            lq = lq[batch_indices, channel_indices, i, j]

            top_gt, left_gt = top * self.args.scale, left * self.args.scale
            # Create a meshgrid for the patches
            gt_i = torch.arange(self.args.gt_size, device=gt.device).view(1, 1, -1, 1)
            gt_j = torch.arange(self.args.gt_size, device=gt.device).view(1, 1, 1, -1)

            # Offset the indices by the top-left corner positions
            gt_i = gt_i + top_gt.view(-1, 1, 1, 1)
            gt_j = gt_j + left_gt.view(-1, 1, 1, 1)

            gt = gt[batch_indices, channel_indices, gt_i, gt_j]
            
            x_pos = ((i.expand(-1, -1, -1, lq_size) + 0.5) / h_lq - 0.5) * 2
            y_pos = ((j.expand(-1, -1, lq_size, -1) + 0.5) / w_lq - 0.5) * 2
            lq_pos = torch.cat((x_pos, y_pos), dim=1)

            global_lq_scale = self.args.get('global_lq_scale', 1.0)
            global_lq = F.interpolate(global_lq, size=(int(global_lq_scale * lq.size(-2)), int(global_lq_scale * lq.size(-1))), mode='bicubic')

            global_lq_x_pos = ((torch.arange(global_lq.shape[-2], device=global_lq.device) + 0.5) / global_lq.shape[-2] - 0.5) * 2
            global_lq_y_pos = ((torch.arange(global_lq.shape[-1], device=global_lq.device) + 0.5) / global_lq.shape[-1] - 0.5) * 2

            global_lq_x_pos = global_lq_x_pos.view(1, 1, -1, 1).expand(global_lq.shape[0], -1, -1, global_lq.shape[-1])
            global_lq_y_pos = global_lq_y_pos.view(1, 1, 1, -1).expand(global_lq.shape[0], -1, global_lq.shape[-2], -1)

            global_lq_pos = torch.cat((global_lq_x_pos, global_lq_y_pos), dim=1)

            results = {'lq': lq, 'gt': gt, 'global_lq': global_lq, 'lq_pos': lq_pos, 'global_lq_pos': global_lq_pos}

            for k, v in data.items():
                if 'input_ids' in k or 'is_null_captions' == k:
                    results[k] = v
                            
            if self.queue_size > 0:
                results = self._dequeue_and_enqueue(results)
            
            results['lq'] = results['lq'].contiguous()
            return results
    
        # random crop
        gt_size = self.args.gt_size
        gt, lq = paired_random_crop(gt, lq, gt_size, self.args.scale)

        if self.args.resize_lq_to_hq:
            if random.random() < self.args.no_degradation_prob or torch.isnan(lq).any():
                lq = gt
            else:
                lq = F.interpolate(lq, size=(gt.size(-2), gt.size(-1)), mode='bicubic')

        results = {'lq': lq, 'gt': gt}

        for k, v in data.items():
            if 'input_ids' in k or 'is_null_captions' == k or 'latent' in k:
                results[k] = v
                        
        if self.queue_size > 0:
            results = self._dequeue_and_enqueue(results)
        
        results['lq'] = results['lq'].contiguous()
        return results

        # if 'input_ids' in data:
        #     # training pair pool
        #     if self.queue_size > 0:
        #         lq, gt, input_ids = self._dequeue_and_enqueue(lq, gt, data['input_ids'])
        #     else:
        #         lq, gt, input_ids = lq, gt, data['input_ids']
        #     lq = lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

        #     return lq, gt, input_ids
        
        # # training pair pool
        # if self.queue_size > 0:
        #     lq, gt = self._dequeue_and_enqueue(lq, gt)
        # lq = lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

        # return lq, gt


class RealESRGANDegradationTwoLQModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp()  # do usm sharpening
        self.queue_size = args.queue_size
        self.queue_ptr = 0

        self.args = args

    @torch.no_grad()
    def _dequeue_and_enqueue(self, data):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        if not hasattr(self, 'queue_lr'):
            for k, v in data.items():
                if not hasattr(self, f'queue_{k}'):
                    assert self.queue_size % v.size(0) == 0, f'queue size {self.queue_size} should be divisible by batch size {v.size(0)}'
                    setattr(self, f'queue_{k}', torch.zeros(self.queue_size, *v.size()[1:], dtype=v.dtype, device=v.device))
            
        if self.queue_ptr == self.queue_size:
            # shuffle
            idx = torch.randperm(self.queue_size)
            results = {}
            for k, v in data.items():
                setattr(self, f'queue_{k}', getattr(self, f'queue_{k}')[idx])

                # get first b samples
                results[k] = getattr(self, f'queue_{k}')[0:v.size(0)].clone()
                # update the queue
                getattr(self, f'queue_{k}')[0:v.size(0)] = v.clone()
            return results
        else:
            for k, v in data.items():
                getattr(self, f'queue_{k}')[self.queue_ptr:self.queue_ptr + v.size(0)] = v.clone()
            self.queue_ptr = self.queue_ptr + v.size(0)
            return data

    @torch.no_grad()
    def __call__(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # training data synthesis
        gt = data['gt']
        if self.args.use_usm:
            gt = self.usm_sharpener(gt)

        ori_h, ori_w = gt.size()[2:4]

        # ----------------------- generate student lq ----------------------- #
        student_kernel1 = data['student_kernel1']
        student_kernel2 = data['student_kernel2']
        student_sinc_kernel = data['student_sinc_kernel']


        # ----------------------- The first degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            student_lq = filter2D(gt, student_kernel1)
        else:
            student_lq = gt

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.args.student_args.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.args.student_args.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.args.student_args.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        student_lq = F.interpolate(student_lq, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            gray_noise_prob = self.args.student_args.gray_noise_prob
            if np.random.uniform() < self.args.student_args.gaussian_noise_prob:
                student_lq = random_add_gaussian_noise_pt(
                    student_lq, sigma_range=self.args.student_args.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                student_lq = random_add_poisson_noise_pt(
                    student_lq,
                    scale_range=self.args.student_args.poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
        # JPEG compression
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            jpeg_p = student_lq.new_zeros(student_lq.size(0)).uniform_(*self.args.student_args.jpeg_range)
            student_lq = torch.clamp(student_lq, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifactss
            student_lq = self.jpeger(student_lq, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            if np.random.uniform() < self.args.student_args.second_blur_prob:
                student_lq = filter2D(student_lq, student_kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.args.student_args.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.args.student_args.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.args.student_args.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        student_lq = F.interpolate(
            student_lq, size=(int(ori_h / self.args.student_args.scale * scale), int(ori_w / self.args.student_args.scale * scale)), mode=mode)
        # add noise
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            gray_noise_prob = self.args.student_args.gray_noise_prob2
            if np.random.uniform() < self.args.student_args.gaussian_noise_prob2:
                student_lq = random_add_gaussian_noise_pt(
                    student_lq, sigma_range=self.args.student_args.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                student_lq = random_add_poisson_noise_pt(
                    student_lq,
                    scale_range=self.args.student_args.poisson_scale_range2,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            student_lq = F.interpolate(student_lq, size=(ori_h // self.args.student_args.scale, ori_w // self.args.student_args.scale), mode=mode)
            if np.random.uniform() < self.args.student_args.get('gating_p', 1):
                student_lq = filter2D(student_lq, student_sinc_kernel)
            student_lq = torch.clamp(student_lq, 0, 1)
            # JPEG compression
            if np.random.uniform() < self.args.student_args.get('gating_p', 1):
                jpeg_p = student_lq.new_zeros(student_lq.size(0)).uniform_(*self.args.student_args.jpeg_range2)
                student_lq = self.jpeger(student_lq, quality=jpeg_p)
        else:
            student_lq = torch.clamp(student_lq, 0, 1)
            # JPEG compression
            if np.random.uniform() < self.args.student_args.get('gating_p', 1):
                jpeg_p = student_lq.new_zeros(student_lq.size(0)).uniform_(*self.args.student_args.jpeg_range2)
                student_lq = self.jpeger(student_lq, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            student_lq = F.interpolate(student_lq, size=(ori_h // self.args.student_args.scale, ori_w // self.args.student_args.scale), mode=mode)
            if np.random.uniform() < self.args.student_args.get('gating_p', 1):
                student_lq = filter2D(student_lq, student_sinc_kernel)

        # clamp and round
        student_lq = torch.clamp((student_lq * 255.0).round(), 0, 255) / 255.

        if self.args.student_args.resize_lq_to_hq:
            if random.random() < self.args.student_args.no_degradation_prob or torch.isnan(student_lq).any():
                student_lq = gt
            else:
                student_lq = F.interpolate(student_lq, size=(gt.size(-2), gt.size(-1)), mode='bicubic')


        # ----------------------- generate teacher lq ----------------------- #
        teacher_kernel1 = data['teacher_kernel1']
        teacher_kernel2 = data['teacher_kernel2']
        teacher_sinc_kernel = data['teacher_sinc_kernel']

        # ----------------------- The first degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
            teacher_lq = filter2D(gt, teacher_kernel1)
        else:
            teacher_lq = gt

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.args.teacher_args.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.args.teacher_args.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.args.teacher_args.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        teacher_lq = F.interpolate(teacher_lq, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
            gray_noise_prob = self.args.teacher_args.gray_noise_prob
            if np.random.uniform() < self.args.teacher_args.gaussian_noise_prob:
                teacher_lq = random_add_gaussian_noise_pt(
                    teacher_lq, sigma_range=self.args.teacher_args.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                teacher_lq = random_add_poisson_noise_pt(
                    teacher_lq,
                    scale_range=self.args.teacher_args.poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
        # JPEG compression
        if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
            jpeg_p = teacher_lq.new_zeros(teacher_lq.size(0)).uniform_(*self.args.teacher_args.jpeg_range)
            teacher_lq = torch.clamp(teacher_lq, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifactss
            teacher_lq = self.jpeger(teacher_lq, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
            if np.random.uniform() < self.args.teacher_args.second_blur_prob:
                teacher_lq = filter2D(teacher_lq, teacher_kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.args.teacher_args.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.args.teacher_args.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.args.teacher_args.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        teacher_lq = F.interpolate(
            teacher_lq, size=(int(ori_h / self.args.teacher_args.scale * scale), int(ori_w / self.args.teacher_args.scale * scale)), mode=mode)
        # add noise
        if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
            gray_noise_prob = self.args.teacher_args.gray_noise_prob2
            if np.random.uniform() < self.args.teacher_args.gaussian_noise_prob2:
                teacher_lq = random_add_gaussian_noise_pt(
                    teacher_lq, sigma_range=self.args.teacher_args.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                teacher_lq = random_add_poisson_noise_pt(
                    teacher_lq,
                    scale_range=self.args.teacher_args.poisson_scale_range2,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            teacher_lq = F.interpolate(teacher_lq, size=(ori_h // self.args.teacher_args.scale, ori_w // self.args.teacher_args.scale), mode=mode)
            if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
                teacher_lq = filter2D(teacher_lq, teacher_sinc_kernel)
            teacher_lq = torch.clamp(teacher_lq, 0, 1)
            # JPEG compression
            if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
                jpeg_p = teacher_lq.new_zeros(teacher_lq.size(0)).uniform_(*self.args.teacher_args.jpeg_range2)
                teacher_lq = self.jpeger(teacher_lq, quality=jpeg_p)
        else:
            teacher_lq = torch.clamp(teacher_lq, 0, 1)
            # JPEG compression
            if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
                jpeg_p = teacher_lq.new_zeros(teacher_lq.size(0)).uniform_(*self.args.teacher_args.jpeg_range2)
                teacher_lq = self.jpeger(teacher_lq, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            teacher_lq = F.interpolate(teacher_lq, size=(ori_h // self.args.teacher_args.scale, ori_w // self.args.teacher_args.scale), mode=mode)
            if np.random.uniform() < self.args.teacher_args.get('gating_p', 1):
                teacher_lq = filter2D(teacher_lq, teacher_sinc_kernel)

        # clamp and round
        teacher_lq = torch.clamp((teacher_lq * 255.0).round(), 0, 255) / 255.

        if self.args.teacher_args.resize_lq_to_hq:
            if random.random() < self.args.teacher_args.no_degradation_prob or torch.isnan(teacher_lq).any():
                teacher_lq = gt
            else:
                teacher_lq = F.interpolate(teacher_lq, size=(gt.size(-2), gt.size(-1)), mode='bicubic')

        # random crop
        # we skip random crop by assuming gt is gt_size
        # gt_size = self.args.gt_size
        # gt, lq = paired_random_crop(gt, lq, gt_size, self.args.scale)

        results = {'lq': student_lq,
                   'teacher_lq': teacher_lq,
                   'gt': gt}
        
        for k, v in data.items():
            if 'input_ids' in k or 'is_null_captions' == k:
                results[k] = v

        if self.queue_size > 0:
            results = self._dequeue_and_enqueue(results)
        
        results['lq'] = results['lq'].contiguous()
        return results


class RealESRGANDegradationTwoLQModelv2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp()  # do usm sharpening
        self.queue_size = args.queue_size

        self.args = args

    @torch.no_grad()
    def _dequeue_and_enqueue(self, student_lq, teacher_lq, gt, input_ids=None):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = student_lq.size()
        if not hasattr(self, 'queue_student_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'

            self.queue_student_lr = torch.zeros(self.queue_size, c, h, w, dtype=student_lq.dtype, device=student_lq.device)

            _, c, h, w = teacher_lq.size()
            self.queue_teacher_lr = torch.zeros(self.queue_size, c, h, w, dtype=teacher_lq.dtype, device=teacher_lq.device)

            _, c, h, w = gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w, dtype=gt.dtype, device=gt.device)

            if input_ids is not None:
                _, n = input_ids.size()
                self.queue_input_ids = torch.zeros(self.queue_size, n, dtype=input_ids.dtype, device=input_ids.device)
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_student_lr = self.queue_student_lr[idx]
            self.queue_teacher_lr = self.queue_teacher_lr[idx]
            self.queue_gt = self.queue_gt[idx]

            # get first b samples
            student_lq_dequeue = self.queue_student_lr[0:b, :, :, :].clone()
            teacher_lq_dequeue = self.queue_teacher_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_student_lr[0:b, :, :, :] = student_lq.clone()
            self.queue_teacher_lr[0:b, :, :, :] = teacher_lq.clone()
            self.queue_gt[0:b, :, :, :] = gt.clone()
            if input_ids is not None:
                self.queue_input_ids = self.queue_input_ids[idx]
                # get first b samples
                input_ids_dequeue = self.queue_input_ids[0:b, :].clone()
                # update the queue
                self.queue_input_ids[0:b, :] = input_ids.clone()

                return student_lq_dequeue, teacher_lq_dequeue, gt_dequeue, input_ids_dequeue

            return student_lq_dequeue, teacher_lq_dequeue, gt_dequeue
        else:
            # only do enqueue
            self.queue_student_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = student_lq.clone()
            self.queue_teacher_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = teacher_lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = gt.clone()
            if input_ids is not None:
                self.queue_input_ids[self.queue_ptr:self.queue_ptr + b, :] = input_ids.clone()
            self.queue_ptr = self.queue_ptr + b

            if input_ids is not None:
                return student_lq, teacher_lq, gt, input_ids
            
            return student_lq, teacher_lq, gt

    @torch.no_grad()
    def __call__(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # training data synthesis
        gt = data['gt']
        if self.args.use_usm:
            gt = self.usm_sharpener(gt)

        ori_h, ori_w = gt.size()[2:4]

        # ----------------------- generate student lq ----------------------- #
        student_kernel1 = data['student_kernel1']
        student_kernel2 = data['student_kernel2']
        student_sinc_kernel = data['student_sinc_kernel']


        # ----------------------- The first degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            student_lq = filter2D(gt, student_kernel1)
        else:
            student_lq = gt

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.args.student_args.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.args.student_args.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.args.student_args.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        student_lq = F.interpolate(student_lq, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            gray_noise_prob = self.args.student_args.gray_noise_prob
            if np.random.uniform() < self.args.student_args.gaussian_noise_prob:
                student_lq = random_add_gaussian_noise_pt(
                    student_lq, sigma_range=self.args.student_args.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                student_lq = random_add_poisson_noise_pt(
                    student_lq,
                    scale_range=self.args.student_args.poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
        # JPEG compression
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            jpeg_p = student_lq.new_zeros(student_lq.size(0)).uniform_(*self.args.student_args.jpeg_range)
            student_lq = torch.clamp(student_lq, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifactss
            student_lq = self.jpeger(student_lq, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            if np.random.uniform() < self.args.student_args.second_blur_prob:
                student_lq = filter2D(student_lq, student_kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.args.student_args.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.args.student_args.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.args.student_args.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        student_lq = F.interpolate(
            student_lq, size=(int(ori_h / self.args.student_args.scale * scale), int(ori_w / self.args.student_args.scale * scale)), mode=mode)
        # add noise
        if np.random.uniform() < self.args.student_args.get('gating_p', 1):
            gray_noise_prob = self.args.student_args.gray_noise_prob2
            if np.random.uniform() < self.args.student_args.gaussian_noise_prob2:
                student_lq = random_add_gaussian_noise_pt(
                    student_lq, sigma_range=self.args.student_args.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                student_lq = random_add_poisson_noise_pt(
                    student_lq,
                    scale_range=self.args.student_args.poisson_scale_range2,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            student_lq = F.interpolate(student_lq, size=(ori_h // self.args.student_args.scale, ori_w // self.args.student_args.scale), mode=mode)
            teacher_lq = F.interpolate(teacher_lq, size=(ori_h // self.args.teacher_args.scale, ori_w // self.args.teacher_args.scale), mode=mode)
            if np.random.uniform() < self.args.student_args.get('gating_p', 1):
                student_lq = filter2D(student_lq, student_sinc_kernel)
                teacher_lq = filter2D(teacher_lq, student_sinc_kernel)
            student_lq = torch.clamp(student_lq, 0, 1)
            teacher_lq = torch.clamp(teacher_lq, 0, 1)
            # JPEG compression
            if np.random.uniform() < self.args.student_args.get('gating_p', 1):
                jpeg_p = student_lq.new_zeros(student_lq.size(0)).uniform_(*self.args.student_args.jpeg_range2)
                student_lq = self.jpeger(student_lq, quality=jpeg_p)
                teacher_lq = self.jpeger(teacher_lq, quality=jpeg_p)
        else:
            student_lq = torch.clamp(student_lq, 0, 1)
            # JPEG compression
            if np.random.uniform() < self.args.student_args.get('gating_p', 1):
                jpeg_p = student_lq.new_zeros(student_lq.size(0)).uniform_(*self.args.student_args.jpeg_range2)
                student_lq = self.jpeger(student_lq, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            student_lq = F.interpolate(student_lq, size=(ori_h // self.args.student_args.scale, ori_w // self.args.student_args.scale), mode=mode)
            teacher_lq = F.interpolate(teacher_lq, size=(ori_h // self.args.teacher_args.scale, ori_w // self.args.teacher_args.scale), mode=mode)
            if np.random.uniform() < self.args.student_args.get('gating_p', 1):
                student_lq = filter2D(student_lq, student_sinc_kernel)
                teacher_lq = filter2D(teacher_lq, student_sinc_kernel)

        # clamp and round
        student_lq = torch.clamp((student_lq * 255.0).round(), 0, 255) / 255.
        teacher_lq = torch.clamp((teacher_lq * 255.0).round(), 0, 255) / 255.

        if self.args.student_args.resize_lq_to_hq:
            if random.random() < self.args.student_args.no_degradation_prob or torch.isnan(student_lq).any():
                student_lq = gt
            else:
                student_lq = F.interpolate(student_lq, size=(gt.size(-2), gt.size(-1)), mode='bicubic')

            if random.random() < self.args.student_args.no_degradation_prob or torch.isnan(teacher_lq).any():
                teacher_lq = gt
            else:
                teacher_lq = F.interpolate(teacher_lq, size=(gt.size(-2), gt.size(-1)), mode='bicubic')


        # random crop
        # we skip random crop by assuming gt is gt_size
        # gt_size = self.args.gt_size
        # gt, lq = paired_random_crop(gt, lq, gt_size, self.args.scale)

        if 'input_ids' in data:
            # training pair pool
            if self.queue_size > 0:
                student_lq, teacher_lq, gt, input_ids = self._dequeue_and_enqueue(student_lq, teacher_lq, gt, data['input_ids'])
            else:
                student_lq, teacher_lq, gt, input_ids = student_lq, teacher_lq, gt, data['input_ids']
            student_lq, teacher_lq = student_lq.contiguous(), teacher_lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return student_lq, teacher_lq, gt, input_ids
        
        # training pair pool
        if self.queue_size > 0:
            student_lq, teacher_lq, gt = self._dequeue_and_enqueue(student_lq, teacher_lq, gt)
        student_lq, teacher_lq = student_lq.contiguous(), teacher_lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

        return student_lq, teacher_lq, gt
    
    
class BicubicDegradationModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

    @torch.no_grad()
    def __call__(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # training data synthesis
        gt = data['gt']
        out = F.interpolate(gt, size=(gt.size(-2) // self.args.scale, gt.size(-1) // self.args.scale), mode='bicubic')

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # random crop
        gt_size = self.args.gt_size
        gt, lq = paired_random_crop(gt, lq, gt_size, self.args.scale)

        if self.args.resize_lq_to_hq:
            if random.random() < self.args.no_degradation_prob or torch.isnan(lq).any():
                lq = gt
            else:
                lq = F.interpolate(lq, size=(gt.size(-2), gt.size(-1)), mode='bicubic')

        if 'input_ids' in data:
            return lq, gt, data['input_ids']
        
        return lq, gt