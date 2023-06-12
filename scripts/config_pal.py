from os import path

comfy_url = 'http://127.0.0.1:8188/'
comfy_root = '../../ComfyUI'
ckpt_name = 'RetroDiffusionModelV3.safetensors'
palette_name = 'NES'
model_shortname = 'rd.' + palette_name

repo_root = path.join(path.dirname(__file__), '..')
assets_dir = path.join(repo_root, 'work', f'out_learn_{model_shortname}')
weights_dir = path.join(repo_root, 'approx_vae')

latent_size = [64, 64]
