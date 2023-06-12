from enum import Enum, auto
import fnmatch
import torch
from os import listdir, makedirs
from os.path import join, dirname, exists
from torch import Tensor, IntTensor, FloatTensor, inference_mode, load, save
from torch.optim import AdamW
from helpers.device import get_device_type, DeviceLiteral
from helpers.approx_vae.decoder import Decoder
from helpers.approx_vae.dataset import get_data, Dataset
from helpers.approx_vae.get_file_names import GetFileNames
from helpers.approx_vae.get_latents import get_latents
from helpers.approx_vae.resize_samples import get_resized_samples
from helpers.palette import load_palette
import numpy as np
from PIL import Image
from torch.nn import CrossEntropyLoss

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

from config import palette_name, model_shortname, assets_dir, weights_dir

palette = load_palette(palette_name)
# note only the number of colors is used by the model training, as color indexes are used as-is from the PNG input images
# (so they need to be consistent)
# the actual palette is only used to write the test image
samples_dir=join(assets_dir, 'png')
processed_train_data_dir=join(assets_dir, 'processed_train_data')
processed_test_data_dir=join(assets_dir, 'processed_test_data')
latents_dir=join(assets_dir, 'pt')
test_latents_dir=join(assets_dir, 'test_pt')
test_samples_dir=join(assets_dir, 'test_png')
predictions_dir=join(assets_dir, 'test_pred5')
for path_ in [processed_train_data_dir, processed_test_data_dir, predictions_dir]:
  makedirs(path_, exist_ok=True)

weights_path = join(weights_dir, f'decoder_{model_shortname}.pt')

class Mode(Enum):
  Train = auto()
  Test = auto()
mode = Mode.Train
#mode = Mode.Test
test_after_train=True
resume_training=False

model = Decoder(len(palette) // 3)
if exists(weights_path) and resume_training or mode is not Mode.Train:
  model.load_state_dict(load(weights_path, weights_only=True))
model = model.to(device)

training_dtype = torch.float32

epochs = 10000
# epochs = 1000
# epochs = 400

optim = AdamW(model.parameters(), lr=5e-2)
loss_fn = CrossEntropyLoss()

def train(epoch: int, dataset: Dataset):
  model.train()
  out: Tensor = model(dataset.latents)
  loss = loss_fn(out, dataset.samples)
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    print(f'epoch {epoch:04d}, {loss:.3f}')

@inference_mode(True)
def test(palette):
  get_latent_filenames: GetFileNames = lambda: sorted(fnmatch.filter(listdir(test_latents_dir), f"*.pt"))
  get_sample_filenames: GetFileNames = lambda: [latent_path.replace('pt', 'png') for latent_path in get_latent_filenames()]

  latents: FloatTensor = get_latents(
    test_latents_dir,
    processed_test_data_dir,
    get_latent_filenames,
    device=device,
  )
  latents = latents.to(training_dtype)

  true_samples: IntTensor = get_resized_samples(
    in_dir=test_samples_dir,
    out_dir=processed_test_data_dir,
    get_sample_filenames=get_sample_filenames,
    device=device,
  )
  true_samples = true_samples.to(torch.long) # need longs for CrossEntropyLoss

  model.eval() # might be redundant due to inference mode but whatever

  predicts: Tensor = model.forward(latents)

  loss = loss_fn(predicts, true_samples)
  print(f'validation {loss:.3f}')

  predicts = torch.argmax(predicts, 1) # maximum score
  predicts = predicts.cpu().numpy()
  predicts = predicts.astype(np.uint8)

  for prediction, sample_filename in zip(predicts, get_sample_filenames()):
    img = Image.fromarray(prediction)
    img.putpalette(palette)
    img.save(join(predictions_dir, sample_filename))


match(mode):
  case Mode.Train:
    dataset: Dataset = get_data(
      latents_dir=latents_dir,
      processed_train_data_dir=processed_train_data_dir,
      samples_dir=samples_dir,
      dtype=training_dtype,
      device=device,
    )
    for epoch in range(epochs):
      train(epoch, dataset)
    del dataset
    save(model.state_dict(), weights_path)
    if test_after_train:
      test(palette)
  case Mode.Test:
    test(palette)
