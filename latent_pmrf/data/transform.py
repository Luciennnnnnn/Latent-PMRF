import math
import random

from PIL import Image
import imageio

try:
    import rawpy
except ValueError:
    pass

import numpy as np

import torch

from torch.utils import data as data
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from .augmentation import paired_random_crop, augment
from .degradation import circular_lowpass_kernel, random_mixed_kernels


pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
pulse_tensor[10, 10] = 1


def generate_kernels(args):
    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(args.kernel_range)
    kernel_size1 = kernel_size
    if np.random.uniform() < args.sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            args.kernel_list,
            args.kernel_prob,
            kernel_size,
            args.blur_sigma,
            args.blur_sigma, [-math.pi, math.pi],
            args.betag_range,
            args.betap_range,
            noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(args.kernel_range)
    kernel_size2 = kernel_size

    # print(f"{AcceleratorState().process_index=} {kernel_size1=} {kernel_size2=}", flush=True)
    if np.random.uniform() < args.sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            args.kernel_list2,
            args.kernel_prob2,
            kernel_size,
            args.blur_sigma2,
            args.blur_sigma2, [-math.pi, math.pi],
            args.betag_range2,
            args.betap_range2,
            noise_range=None)

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < args.final_sinc_prob:
        kernel_size = random.choice(args.kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)

    return kernel, kernel2, sinc_kernel


def tokenize_captions(texts, tokenizer, args, is_train=True):
    captions = []
    is_null_captions = []
    for i, caption in enumerate(texts):
        if random.random() < args.get('proportion_empty_prompts', 1):
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
            raise NotImplementedError("???")
        
    inputs = tokenizer(
        captions, max_length=args.get('max_tokens', tokenizer.model_max_length), padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids, torch.tensor(is_null_captions, dtype=torch.bool)

def tokenize_captions_sd3(texts, tokenizer_one, tokenizer_two, tokenizer_three, args, is_train=True):
    captions = []
    is_null_captions = []
    for i, caption in enumerate(texts):
        if random.random() < args.get('proportion_empty_prompts', 1):
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
            raise NotImplementedError("???")
        
    inputs_one = tokenizer_one(
        captions, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
    )
    inputs_two = tokenizer_two(
        captions, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
    )
    inputs_three = tokenizer_three(
        captions, max_length=77, padding="max_length", truncation=True, add_special_tokens=True, return_tensors="pt"
    )
    return inputs_one.input_ids, inputs_two.input_ids, inputs_three.input_ids, torch.tensor(is_null_captions, dtype=torch.bool)


def realesrgan_transform_sd3(examples, tokenizer_one, tokenizer_two, tokenizer_three, args):
    random_crop = transforms.RandomCrop(args.gt_size)

    # from PIL import Image
    # images = [Image.open(image) for image in examples['image']]
    
    images = examples['image']
    images = [random_crop(image.convert("RGB")) for image in images]
    images = [augment(np.array(image), args.use_hflip, args.use_rot) for image in images]
    images = [to_tensor(image) for image in images]

    transformed_examples = {}

    transformed_examples["gt"] = images

    if 'teacher_args' in args:
        student_kernel1, student_kernel2, student_sinc_kernel = zip(*[generate_kernels(args.student_args) for image in images])
        transformed_examples["student_kernel1"] = student_kernel1
        transformed_examples["student_kernel2"] = student_kernel2
        transformed_examples["student_sinc_kernel"] = student_sinc_kernel

        teacher_kernel1, teacher_kernel2, teacher_sinc_kernel = zip(*[generate_kernels(args.teacher_args) for image in images])
        transformed_examples["teacher_kernel1"] = teacher_kernel1
        transformed_examples["teacher_kernel2"] = teacher_kernel2
        transformed_examples["teacher_sinc_kernel"] = teacher_sinc_kernel
    else:
        kernel1, kernel2, sinc_kernel = zip(*[generate_kernels(args) for image in images])
        transformed_examples["kernel1"] = kernel1
        transformed_examples["kernel2"] = kernel2
        transformed_examples["sinc_kernel"] = sinc_kernel

    input_ids_one, input_ids_two, input_ids_three, is_null_captions = tokenize_captions_sd3(examples["text"], tokenizer_one, tokenizer_two, tokenizer_three, args)
    transformed_examples["input_ids_one"] = input_ids_one
    transformed_examples["input_ids_two"] = input_ids_two
    transformed_examples["input_ids_three"] = input_ids_three
    transformed_examples["is_null_captions"] = is_null_captions

    return transformed_examples


def realesrgan_transform(examples, tokenizer, args):
    if 'latent' in examples:
        images = [np.array(image.convert("RGB")) for image in examples['image']]
        latents = [np.transpose(np.array(latent, dtype=np.float16), axes=(1, 2, 0)) for latent in examples['latent']]

        for i in range(len(images)):
            images[i], latents[i] = paired_random_crop(images[i], latents[i], args.gt_size, 8)
            images[i], latents[i] = augment([images[i], latents[i]], args.use_hflip, args.use_rot)
        
        images = [to_tensor(image) for image in images]
        latents = [torch.tensor(np.transpose(latent, axes=(2, 0, 1)).copy(), dtype=torch.float16) for latent in latents]
    else:
        random_crop = transforms.RandomCrop(args.gt_size)
        images = [random_crop(image.convert("RGB")) for image in examples['image']]
        images = [augment(np.array(image), args.use_hflip, args.use_rot) for image in images]
        images = [to_tensor(image) for image in images]

    transformed_examples = {}

    transformed_examples["gt"] = images
    if 'latent' in examples:
        transformed_examples['latent'] = latents

    if 'teacher_args' in args:
        student_kernel1, student_kernel2, student_sinc_kernel = zip(*[generate_kernels(args.student_args) for image in images])
        transformed_examples["student_kernel1"] = student_kernel1
        transformed_examples["student_kernel2"] = student_kernel2
        transformed_examples["student_sinc_kernel"] = student_sinc_kernel

        teacher_kernel1, teacher_kernel2, teacher_sinc_kernel = zip(*[generate_kernels(args.teacher_args) for image in images])
        transformed_examples["teacher_kernel1"] = teacher_kernel1
        transformed_examples["teacher_kernel2"] = teacher_kernel2
        transformed_examples["teacher_sinc_kernel"] = teacher_sinc_kernel
    else:
        kernel1, kernel2, sinc_kernel = zip(*[generate_kernels(args) for image in images])
        transformed_examples["kernel1"] = kernel1
        transformed_examples["kernel2"] = kernel2
        transformed_examples["sinc_kernel"] = sinc_kernel

    input_ids, is_null_captions = tokenize_captions(examples["text"], tokenizer, args)
    transformed_examples["input_ids"] = input_ids
    transformed_examples["is_null_captions"] = is_null_captions

    return transformed_examples


def realesrgan_transform_semantic_segmentation(examples, args):
    images = [np.array(image.convert("RGB")) for image in examples['image']]
    semantic_segmentations = [np.array(semantic_segmentation.convert("RGB")) for semantic_segmentation in examples['semantic_segmentation']]

    for i in range(len(images)):
        images[i], semantic_segmentations[i] = paired_random_crop(images[i], semantic_segmentations[i], args.gt_size, 1)
        
        images[i], semantic_segmentations[i] = augment([images[i], semantic_segmentations[i]], args.use_hflip, args.use_rot)

    images = [to_tensor(image) for image in images]
    semantic_segmentations = [to_tensor(semantic_segmentation) for semantic_segmentation in semantic_segmentations]

    kernel1, kernel2, sinc_kernel = zip(*[generate_kernels(args) for image in images])

    examples["gt"] = images
    examples["semantic_segmentation"] = semantic_segmentations
    examples["kernel1"] = kernel1
    examples["kernel2"] = kernel2
    examples["sinc_kernel"] = sinc_kernel
    return examples


def bayer_unification(bayer_arrays, args):
    if not isinstance(bayer_arrays, list):
        bayer_arrays = [bayer_arrays]
    
    for i in range(len(bayer_arrays)):
        if args.bayer_pattern_out == 'BGGR':
            if args.bayer_pattern_in == 'GRBG':
                bayer_arrays[i] = bayer_arrays[i][1:-1, :, :]
            elif args.bayer_pattern_in == 'GBRG':
                bayer_arrays[i] = bayer_arrays[i][:, 1:-1, :]
            elif args.bayer_pattern_in == 'RGGB':
                bayer_arrays[i] = bayer_arrays[i][1:-1, 1:-1, :]
        elif args.bayer_pattern_out == 'RGGB':
            if args.bayer_pattern_in == 'GRBG':
                bayer_arrays[i] = bayer_arrays[i][:, 1:-1, :]
            elif args.bayer_pattern_in == 'GBRG':
                bayer_arrays[i] = bayer_arrays[i][1:-1, :, :]
            elif args.bayer_pattern_in == 'BGGR':
                bayer_arrays[i] = bayer_arrays[i][1:-1, 1:-1, :]
        else:
            raise NotImplementedError(f"Do not support training bayer pattern {args.bayer_pattern_out}")

    if len(bayer_arrays) == 1:
        bayer_arrays = bayer_arrays[0]
    return bayer_arrays
    

def raw_denoise_transform_backup(examples, tokenizer, args):
    transformed_examples = {}
    transformed_examples['lq'] = []
    transformed_examples['gt'] = []

    lq_raw_path = examples['lq_raw_path']
    gt_raw_path = examples['gt_raw_path']
    ratio = examples['ratio']

    for i in range(len(lq_raw_path)):
        with rawpy.imread(lq_raw_path[i]) as lq_raw, rawpy.imread(gt_raw_path[i]) as gt_raw:
            lq_raw_array = lq_raw.raw_image_visible.copy().astype(np.float32)
            gt_raw_array = gt_raw.raw_image_visible.astype(np.float32)

            lq_raw_array, gt_raw_array = bayer_unification([lq_raw_array, gt_raw_array], args)

            # ------------------------------- random crop ------------------------------- #
            top = random.randint(0, (lq_raw_array.shape[1] - args.gt_size - 1) // 2) * 2
            left = random.randint(0, (lq_raw_array.shape[2] - args.gt_size - 1) // 2) * 2

            lq_raw_array = lq_raw_array[:, top: top + args.gt_size, left: left + args.gt_size]
            gt_raw_array = gt_raw_array[:, top: top + args.gt_size, left: left + args.gt_size]

            # ------------------------------- random flip ------------------------------- #
            # Horizental Flip
            if np.random.uniform() > 0.5:
                lq_raw_array = lq_raw_array[:, ::-1, :][:, 1:-1, :]
                gt_raw_array = gt_raw_array[:, ::-1, :][:, 1:-1, :]
            else:
                if np.random.uniform() > 0.5:
                    lq_raw_array = lq_raw_array[:, 2:, :]
                    gt_raw_array = gt_raw_array[:, 2:, :]
                else:
                    lq_raw_array = lq_raw_array[:, :-2, :]
                    gt_raw_array = gt_raw_array[:, :-2, :]

            # Vertiacal Flip:
            if np.random.uniform() > 0.5:
                lq_raw_array = lq_raw_array[::-1, :, :][1:-1, :, :]
                gt_raw_array = gt_raw_array[::-1, :, :][1:-1, :, :]
            else:
                if np.random.uniform() > 0.5:
                    lq_raw_array = lq_raw_array[2:, :, :]
                    gt_raw_array = gt_raw_array[2:, :, :]
                else:
                    lq_raw_array = lq_raw_array[:-2, :, :]
                    gt_raw_array = gt_raw_array[:-2, :, :]

            # rot90
            if np.random.uniform() > 0.5:
                lq_raw_array = lq_raw_array.transpose(1, 0, 2)
                gt_raw_array = gt_raw_array.transpose(1, 0, 2)
            
            gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

            # normalization
            lq_raw_array = np.maximum(0, lq_raw_array - args.black_level) / (args.white_level - args.black_level)
            gt_rgb = np.float32(gt_rgb / 65535.0)
            
            # scale lq
            lq_raw_array = np.minimum(lq_raw_array * ratio[i], 1.0)

            # to pytorch tensor
            lq_raw_array = torch.from_numpy(lq_raw_array, dtype=torch.float32)
            gt_rgb = torch.from_numpy(gt_rgb, dtype=torch.float32)

            transformed_examples['lq'].append(lq_raw_array)
            transformed_examples['gt'].append(gt_rgb)

    input_ids, is_null_captions = tokenize_captions(examples["text"], tokenizer, args)
    transformed_examples["input_ids"] = input_ids
    transformed_examples["is_null_captions"] = is_null_captions

    return transformed_examples


cache = {}

def raw_denoise_transform(examples, tokenizer, args):

    transformed_examples = {}
    transformed_examples['lq'] = []
    transformed_examples['lq_rgb'] = []
    transformed_examples['gt'] = []

    lq_raw_path = examples['lq_raw_path']
    lq_rgb_path = examples['lq_rgb_path']
    gt_rgb_path = examples['gt_rgb_path']
    ratio = examples['ratio']

    for i in range(len(lq_raw_path)):
        if args.get('in_memory_cache', False):
            if lq_raw_path[i] in cache:
                lq_raw_array = cache[lq_raw_path[i]]
                lq_rgb = cache[lq_rgb_path[i]]
                gt_rgb = cache[gt_rgb_path[i]]
            else:
                with rawpy.imread(lq_raw_path[i]) as lq_raw:
                    lq_raw_array = lq_raw.raw_image_visible.copy().astype(np.float32)[..., None]

                lq_rgb = imageio.imread(lq_rgb_path[i])
                gt_rgb = imageio.imread(gt_rgb_path[i])

                cache[lq_raw_path[i]] = lq_raw_array
                cache[lq_rgb_path[i]] = lq_rgb
                cache[gt_rgb_path[i]] = gt_rgb
        else:
            with rawpy.imread(lq_raw_path[i]) as lq_raw:
                lq_raw_array = lq_raw.raw_image_visible.copy().astype(np.float32)[..., None]

            lq_rgb = imageio.imread(lq_rgb_path[i])
            gt_rgb = imageio.imread(gt_rgb_path[i])

        lq_raw_array = bayer_unification(lq_raw_array, args)

        # ------------------------------- random crop ------------------------------- #
        top = random.randint(0, (lq_raw_array.shape[0] - args.gt_size - 1 - 2) // 2) * 2
        left = random.randint(0, (lq_raw_array.shape[1] - args.gt_size - 1 - 2) // 2) * 2

        # since random flip results in 2 pixel loss, we crop extra 2 pixel in each dimention
        lq_raw_array = lq_raw_array[top: top + args.gt_size + 2, left: left + args.gt_size + 2, :]
        lq_rgb = lq_rgb[top: top + args.gt_size + 2, left: left + args.gt_size + 2, :]
        gt_rgb = gt_rgb[top: top + args.gt_size + 2, left: left + args.gt_size + 2, :]

        # ------------------------------- random flip ------------------------------- #
        # Horizental Flip
        if np.random.uniform() > 0.5:
            lq_raw_array = lq_raw_array[:, ::-1, :][:, 1:-1, :]
            lq_rgb = lq_rgb[:, ::-1, :][:, 1:-1, :]
            gt_rgb = gt_rgb[:, ::-1, :][:, 1:-1, :]
        else:
            if np.random.uniform() > 0.5:
                lq_raw_array = lq_raw_array[:, 2:, :]
                lq_rgb = lq_rgb[:, 2:, :]
                gt_rgb = gt_rgb[:, 2:, :]
            else:
                lq_raw_array = lq_raw_array[:, :-2, :]
                lq_rgb = lq_rgb[:, :-2, :]
                gt_rgb = gt_rgb[:, :-2, :]

        # Vertiacal Flip:
        if np.random.uniform() > 0.5:
            lq_raw_array = lq_raw_array[::-1, :, :][1:-1, :, :]
            lq_rgb = lq_rgb[::-1, :, :][1:-1, :, :]
            gt_rgb = gt_rgb[::-1, :, :][1:-1, :, :]
        else:
            if np.random.uniform() > 0.5:
                lq_raw_array = lq_raw_array[2:, :, :]
                lq_rgb = lq_rgb[2:, :, :]
                gt_rgb = gt_rgb[2:, :, :]
            else:
                lq_raw_array = lq_raw_array[:-2, :, :]
                lq_rgb = lq_rgb[:-2, :, :]
                gt_rgb = gt_rgb[:-2, :, :]

        # rot90
        if np.random.uniform() > 0.5:
            lq_raw_array = lq_raw_array.transpose(1, 0, 2)
            lq_rgb = lq_rgb.transpose(1, 0, 2)
            gt_rgb = gt_rgb.transpose(1, 0, 2)

        # normalization
        lq_raw_array = np.maximum(0, lq_raw_array - args.black_level) / (args.white_level - args.black_level)
        lq_rgb = np.float32(lq_rgb / 65535.0)
        gt_rgb = np.float32(gt_rgb / 65535.0)
        
        # scale lq
        lq_raw_array = np.minimum(lq_raw_array * ratio[i], 1.0)

        # to pytorch tensor
        lq_raw_array = torch.from_numpy(lq_raw_array.transpose(2, 0, 1))
        # pack raw
        # lq_raw_array = torch.nn.functional.pixel_unshuffle(lq_raw_array[None, ...], downscale_factor=2)[0]

        lq_rgb = torch.from_numpy(lq_rgb.transpose(2, 0, 1))
        gt_rgb = torch.from_numpy(gt_rgb.transpose(2, 0, 1))

        transformed_examples['lq'].append(lq_raw_array)
        transformed_examples['lq_rgb'].append(lq_rgb)
        transformed_examples['gt'].append(gt_rgb)

    input_ids, is_null_captions = tokenize_captions(examples["text"], tokenizer, args)
    transformed_examples["input_ids"] = input_ids
    transformed_examples["is_null_captions"] = is_null_captions

    return transformed_examples
