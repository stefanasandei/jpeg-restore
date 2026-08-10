import torch.nn.functional as F


def pad_to_multiple(image, multiple):
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
