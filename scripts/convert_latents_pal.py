# Convert latents to image, store training set for VAE approximation.
import json
from urllib import request, parse
import websocket
import io
import os
import uuid

import torch
import safetensors.torch
from PIL import Image

from helpers.comfy_api import ComfyUIAPI
from helpers.palette import load_palette, make_palette_template_image
from helpers.k_centroid import k_centroid_downscale, k_centroid_downscale_p

prompt = {
    "1": {
        "inputs": {
            "latent": "ComfyUI_00001_.latent"
        },
        "class_type": "LoadLatent",
        "is_changed": [
            "1793ddf774bd81045709ba814236e599c5a6bd7a94721ccb6bbdb8b8bfa2b268"
        ]
    },
    "2": {
        "inputs": {
            "ckpt_name": "RetroDiffusionMegaModelV2.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "3": {
        "inputs": {
            "samples": [
                "1",
                0
            ],
            "vae": [
                "2",
                2
            ]
        },
        "class_type": "VAEDecode"
    },
    "5": {
        "inputs": {
            "images": [
                "3",
                0
            ]
        },
        "class_type": "PreviewImage"
    }
}


def main():
    from config_pal import comfy_url, comfy_root, palette_name, assets_dir, latent_size, ckpt_name
    comfy = ComfyUIAPI(comfy_url)

    latents_in = os.path.join(comfy_root, 'input')
    latents_out = os.path.join(assets_dir, 'pt')
    samples_out = os.path.join(assets_dir, 'png')
    test_latents_out = os.path.join(assets_dir, 'test_pt')
    test_samples_out = os.path.join(assets_dir, 'test_png')
    os.makedirs(latents_out, exist_ok=True)
    os.makedirs(samples_out, exist_ok=True)
    os.makedirs(test_latents_out, exist_ok=True)
    os.makedirs(test_samples_out, exist_ok=True)

    print(f'Loading palette {palette_name}')
    palette = load_palette(palette_name)
    n_colors = len(palette) // 3
    pal_img = make_palette_template_image(palette)
    quant_method = 2
    print(f'Palette has {n_colors} colors in the following order:')
    print(palette)

    # get list of latent files
    latent_filenames = []
    for dirname in os.listdir(latents_in):
        dirname_full = os.path.join(latents_in, dirname)
        if os.path.isdir(dirname_full):
            for filename in os.listdir(dirname_full):
                if filename.endswith('.latent'):
                    latent_filenames.append(dirname + '/' + filename)

    print(f'Found {len(latent_filenames)} latent files')
    filter_size = torch.Size(latent_size)

    for basename in latent_filenames:
        idname = basename.removesuffix('.latent').replace('/', '_')
        print('Processing ', idname)

        ld = safetensors.torch.load_file(os.path.join(latents_in, basename))['latent_tensor']
        if filter_size is not None and ld.shape[2:4] != filter_size: # traning must have all the samme-sized inputs
            continue

        def latent_filename(i):
            if 'test' in idname:
                return os.path.join(test_latents_out, idname + f'_{i}.pt')
            else:
                return os.path.join(latents_out, idname + f'_{i}.pt')
        def sample_filename(i):
            if 'test' in idname:
                return os.path.join(test_samples_out, idname + f'_{i}.png')
            else:
                return os.path.join(samples_out, idname + f'_{i}.png')

        # Skip generation if all image and latent files already exist for this specimen.
        if all(os.path.exists(latent_filename(i)) and os.path.exists(sample_filename(i)) for i in range(len(ld))):
            continue

        prompt['1']['inputs']['latent'] = basename
        prompt['1']['is_changed'] = uuid.uuid4().hex
        prompt['2']['inputs']['ckpt_name'] = ckpt_name
        png_datas = comfy.execute_prompt(prompt)['5']

        assert(len(png_datas) == len(ld))
        for i, (latents, png_data) in enumerate(zip(ld, png_datas)):
            # Write latents
            torch.save(latents, latent_filename(i))

            # Process image
            img = Image.open(io.BytesIO(png_data))
            if quant_method == 1:
                #img = k_centroid_downscale_p(img, palette, method=Image.Quantize.MAXCOVERAGE)
                img = k_centroid_downscale_p(img, palette, method=Image.Quantize.MEDIANCUT)
            elif quant_method == 2:
                img = k_centroid_downscale(img)
                #img = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, kmeans=n_colors, dither=Image.Dither.FLOYDSTEINBERG, palette=pal_img)
                img = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, kmeans=n_colors, dither=Image.Dither.NONE, palette=pal_img)
                #img = img.quantize(colors=n_colors, method=Image.Quantize.MAXCOVERAGE, kmeans=n_colors, dither=Image.Dither.NONE, palette=pal_img)

            # Write image
            img.save(sample_filename(i))
            print(f'  Wrote {os.path.normpath(latent_filename(i))} and {os.path.normpath(sample_filename(i))}')

    comfy.close()

if __name__ == '__main__':
    main()
