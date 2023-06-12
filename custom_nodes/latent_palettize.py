# Mara Huldra 2023
# SPDX-License-Identifier: MIT
'''
Create a palettized image from latents, using pixelvae model.
'''
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

import comfy.model_management
import folder_paths

# EGA palette -- hardcoded for now, should come from the model file
#PALETTE_NAME = 'ega16'
#PALETTE = [0, 0, 0, 0, 0, 170, 0, 170, 0, 0, 170, 170, 170, 0, 0, 170, 0, 170, 170, 85, 0, 170, 170, 170, 85, 85, 85, 85, 85, 255, 85, 255, 85, 85, 255, 255, 255, 85, 85, 255, 85, 255, 255, 255, 85, 255, 255, 255]
# NES palette
PALETTE_NAME = 'NES'
PALETTE = [0, 0, 0, 252, 252, 252, 248, 248, 248, 188, 188, 188, 124, 124, 124, 164, 228, 252, 60, 188, 252, 0, 120, 248, 0, 0, 252, 184, 184, 248, 104, 136, 252, 0, 88, 248, 0, 0, 188, 216, 184, 248, 152, 120, 248, 104, 68, 252, 68, 40, 188, 248, 184, 248, 248, 120, 248, 216, 0, 204, 148, 0, 132, 248, 164, 192, 248, 88, 152, 228, 0, 88, 168, 0, 32, 240, 208, 176, 248, 120, 88, 248, 56, 0, 168, 16, 0, 252, 224, 168, 252, 160, 68, 228, 92, 16, 136, 20, 0, 248, 216, 120, 248, 184, 0, 172, 124, 0, 80, 48, 0, 216, 248, 120, 184, 248, 24, 0, 184, 0, 0, 120, 0, 184, 248, 184, 88, 216, 84, 0, 168, 0, 0, 104, 0, 184, 248, 216, 88, 248, 152, 0, 168, 68, 0, 88, 0, 0, 252, 252, 0, 232, 216, 0, 136, 136, 0, 64, 88, 248, 216, 248, 120, 120, 120]


# Based on "Tiny AutoEncoder for Stable Diffusion"
def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Decoder(n_colors):
    return nn.Sequential(
        conv(4, 64), nn.ReLU(),
        Block(64, 64), conv(64, n_colors),
    )

# XXX cached model, better way to manage this?
_model = None

def load_model(device):
    global _model
    if _model is None:
        _model = Decoder(len(PALETTE) // 3)
        weights_path = folder_paths.get_full_path("vae_approx", f"decoder_rd.{PALETTE_NAME}.pt")
        _model.load_state_dict(torch.load(weights_path, weights_only=True))
        _model.eval()
        _model = _model.to(device)
    return _model, PALETTE


class LatentPalettize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "palettize"

    CATEGORY = "latent"

    def palettize(self, samples):
        device = comfy.model_management.get_torch_device()
        model, palette = load_model(device)
        predicts = model.forward(samples["samples"].to(device))

        predicts = torch.argmax(predicts, 1) # maximum score
        predicts = predicts.cpu().numpy()
        predicts = predicts.astype(np.uint8)

        results = []
        for prediction in predicts:
            img = Image.fromarray(prediction)
            img.putpalette(palette)
            img = img.convert('RGB')
            results.append(np.array(img))

        result = np.array(results).astype(np.float32) / 255.0
        return (torch.from_numpy(result), )


NODE_CLASS_MAPPINGS = {
    "LatentPalettize": LatentPalettize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentPalettize": "LatentPalettize"
}

