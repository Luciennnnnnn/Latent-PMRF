import os

from torch.nn import functional as F


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def search_image_files(data_dir):
    images = []
    for dirpath, _, fnames in os.walk(data_dir):
        for fname in fnames:
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    
    images = sorted(images)
    
    return images


def patchify(input, patch_size, stride):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    B, C, _, _ = input.size()
    patches = input.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1])
    patches = patches.contiguous().view(B, C, -1, patch_size[0], patch_size[1]).permute(0, 2, 1, 3, 4).reshape(-1, C, patch_size[0], patch_size[1])
    return patches


def unpatchify(input, patch_size, stride, H, W, B):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    _, C, _, _ = input.size()
    output = input.view(B, -1, C, patch_size[0] * patch_size[1])
    output = output.permute(0, 2, 3, 1)
    output = output.contiguous().view(B, C * patch_size[0] * patch_size[1], -1)
    return F.fold(output, output_size=(H, W), kernel_size=patch_size, stride=stride)