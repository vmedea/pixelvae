from PIL import Image
from os.path import join
from torch import load, save, from_numpy, stack, IntTensor
from .get_file_names import GetFileNames
from typing import Tuple, List
import torch
import numpy as np

debug_resize=False

# there's so few of these we may as well keep them all resident in memory
def get_samples(
  in_dir: str,
  out_dir: str,
  get_sample_filenames: GetFileNames,
  device: torch.device = torch.device('cpu')
) -> IntTensor:
    sample_filenames: List[str] = get_sample_filenames()

    images = [Image.open(join(in_dir, sample_filename)) for sample_filename in sample_filenames]
    assert(all(image.mode == 'P' for image in images))

    # output categories directly
    out = [np.array(i) for i in images]

    tensors = [from_numpy(i) for i in out]
    tensor: IntTensor = stack(tensors)
    return tensor.to(device)
