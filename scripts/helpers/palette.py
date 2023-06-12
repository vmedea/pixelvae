import os

from PIL import Image

PALETTES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'palettes')
PAL_EXT = '.png'

def get_image_colors(pal_img):
    palette = []
    pal_img = pal_img.convert('RGB')
    used = set()
    for y in range(pal_img.height):
        for x in range(pal_img.width):
            color = tuple(pal_img.getpixel((x, y)))
            if not color in used:
                used.add(color)
                palette.extend(color)
    return palette


def load_palette(name):
    return get_image_colors(Image.open(os.path.join(PALETTES_PATH, name + PAL_EXT)))
