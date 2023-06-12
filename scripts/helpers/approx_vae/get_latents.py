from .get_file_names import GetFileNames
from torch import load, stack, save, FloatTensor
from os.path import join
from typing import List
import torch

# there's so few of these we may as well keep them all resident in memory
def get_latents(
  in_dir: str,
  out_dir: str,
  get_latent_filenames: GetFileNames,
  device: torch.device = torch.device('cpu'),
) -> FloatTensor:
    latents: List[FloatTensor] = [load(join(in_dir, pt), map_location=device, weights_only=True) for pt in get_latent_filenames()]
    latents: FloatTensor = stack(latents)
    return latents
