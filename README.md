# Palettized VAE approximation

## Pre-trained demonstration models

Under `approx_vae` there's two pretrained models:

- `decoder_rd.ega.pt` - EGA16 palette
- `decoder_rd.NES.pt` - NES55 palette

## Training

Requirement for training is an existing ComfyUI installation, recent enough to have `LoadLatent` and `SaveLatent` nodes.

Training consists of these phases:

- Edit `scripts/config.py` as needed. Put palettes in `palettes/` as needed.

- Collect `.latent` files. This can be saved with the `_for_testing/SaveLatent` node in ComfyUI. All the latents need to be the same size.

- Put the latent files in subdirectories under `ComfyUI/input`. For example:

```
set1_train/0001.latent
set1_train/0002.latent
...
set1_test/0001.latent
set1_test/0002.latent
...
```

Names can be anything, but they need to be one subdirectory deep and end with `.latent`. If there is `test` anywhere in the name they will be used for the test data instead of the train data.

- Run `scripts/convert_latents_pal.py`. ComfyUI needs to be running for this step. This will automatically run a ComfyUI pipeline that loads each latent, and runs them through VAEDecode. Then it will scale and palettize the images. Both the images and latents are saved to the work directory. It will notice when output files already exist and not overwrite them,
so if you want to start from scratch you need to delete the entire work directory.

- Run `scripts/train_approx_decoder.py`. This will train the model and write it to `approx_vae/`.

## ComfyUI custom node

A ComfyUI custom node is provided in this repository to use the generated model with ComfyUI. Note that it currently hardcodes a palette.

- Make sure you're using a recent-enough checkout of ComfyUI that has a `ComfyUI/models/vae_approx` directory.

- Copy node implementation `custom_nodes/*.py` into `ComfyUI/custom_nodes`.

- Copy models `approx_vae/*.pt` into `ComfyUI/models/vae_approx`.

- Restart ComfyUI (check for no errors on startup).

You should be able to find the 'latent/LatentPalettize' node.

# Credits

`pixelvae` by Mara Huldra, 2023, is released under the MIT license.

Some of the code in this repository was based on:

- [taesd](https://github.com/madebyollin/taesd) "Tiny AutoEncoder for Stable Diffusion" by Ollin Boer Bohan.
- https://github.com/Birch-san/diffusers-play by Birch-san.
- The implementation currently assumes that astropulse's Retro-Diffusion model is used. YMMV with other models.
- Most of the palettes in `palettes/` come from [sd-palettize](https://github.com/Astropulse/sd-palettize).
