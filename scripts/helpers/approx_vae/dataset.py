from dataclasses import dataclass
from torch import FloatTensor, IntTensor
import torch
import fnmatch
from os import listdir
from .get_file_names import GetFileNames
from .get_latents import get_latents
from .resize_samples import get_resized_samples

@dataclass
class Dataset:
  latents: FloatTensor
  samples: FloatTensor

def get_data(
  latents_dir: str,
  processed_train_data_dir: str,
  samples_dir: str,
  dtype: torch.dtype = torch.float32,
  device: torch.device = torch.device('cpu'),
) -> Dataset:
  get_latent_filenames: GetFileNames = lambda: fnmatch.filter(listdir(latents_dir), f"*.pt")
  get_sample_filenames: GetFileNames = lambda: [latent_path.replace('pt', 'png') for latent_path in get_latent_filenames()]

  latents: FloatTensor = get_latents(
    in_dir=latents_dir,
    out_dir=processed_train_data_dir,
    get_latent_filenames=get_latent_filenames,
    device=device,
  )
  latents = latents.to(dtype)

  samples: IntTensor = get_resized_samples(
    in_dir=samples_dir,
    out_dir=processed_train_data_dir,
    get_sample_filenames=get_sample_filenames,
    device=device,
  )
  samples = samples.to(torch.long) # need longs for CrossEntropyLoss
  return Dataset(
    latents=latents,
    samples=samples,
  )
