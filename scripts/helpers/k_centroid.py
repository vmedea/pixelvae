from itertools import product

from PIL import Image


def k_centroid_downscale(image, method=Image.Quantize.MAXCOVERAGE, factor=8, centroids=2):
    '''k-centroid scaling, based on: https://github.com/Astropulse/stable-diffusion-aseprite/blob/main/scripts/image_server.py.'''

    width = image.size[0] // factor
    height = image.size[1] // factor

    out = Image.new('RGB', (width, height))

    for x, y in product(range(width), range(height)):
        tile = image.crop((x * factor, y * factor, (x + 1) * factor, (y + 1) * factor))
        tile = tile.quantize(colors=centroids, method=method, kmeans=centroids, dither=Image.Dither.NONE)
        color_counts = tile.getcolors()
        most_common_idx = max(color_counts, key=lambda x: x[0])[1]
        out.putpixel((x, y), tuple(tile.getpalette()[most_common_idx*3:(most_common_idx + 1)*3]))

    return out


def k_centroid_downscale_p(image, palette, method=Image.Quantize.MAXCOVERAGE, factor=8):
    '''k-centroid downscale with fixed palette'''
    width = image.size[0] // factor
    height = image.size[1] // factor

    out = Image.new('P', (width, height))
    centroids = len(palette) // 3
    out.putpalette(palette)

    for x, y in product(range(width), range(height)):
        tile = image.crop((x * factor, y * factor, (x + 1) * factor, (y + 1) * factor))
        tile = tile.quantize(colors=centroids, method=method, kmeans=centroids, dither=Image.Dither.NONE, palette=out)
        color_counts = tile.getcolors()
        most_common_idx = max(color_counts, key=lambda x: x[0])[1]
        out.putpixel((x, y), most_common_idx)

    return out
