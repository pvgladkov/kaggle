import StringIO

import numpy as np
from PIL import Image, ImageEnhance
from skimage import exposure


def transform_im(im, crop_type=None, alter_type=None, rotate_angle=None):
    alter_types = {
        'resize05': lambda x: resize(x, 0.5),
        'resize08': lambda x: resize(x, 0.8),
        'resize15': lambda x: resize(x, 1.5),
        'resize20': lambda x: resize(x, 2.0),
        'gamma08': lambda x: gamma(x, 0.8),
        'gamma12': lambda x: gamma(x, 1.2),
        'q70': lambda x: jpeg_comp(x, 70),
        'q90': lambda x: jpeg_comp(x, 90)
    }

    if crop_type is not None:
        im = crop(im, crop_type)

    if rotate_angle > 0:
        im = rotate(im, rotate_angle)

    im = alter_types.get(alter_type, lambda x: x)(im)
    return im


def read_and_resize(f_path):
    im_array = np.array(Image.open(f_path), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    return new_array / 255.0


def resize_shape(im, w, h):
    return im.resize((w, h))


def resize(im, scale):
    w, h = im.size
    return im.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)


def gamma(im, g):
    return ImageEnhance.Brightness(im).enhance(g)


def gamma2(im, g):
    im = np.array(im)
    return exposure.adjust_gamma(im, gamma=g)


def crop(im, t):
    w, h = im.size
    crop_types = [
        (w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256),
        (0, 0, 512, 512),
        (w - 512, 0, w, 512),
        (0, h - 512, 512, h),
        (w - 512, h - 512, w, h),
    ]
    return im.crop(crop_types[t])


def rotate(im, angle):
    return im.rotate(angle)


def jpeg_comp(im, q):
    buf = StringIO.StringIO()
    im.save(buf, "JPEG", quality=q)
    img = Image.open(StringIO.StringIO(buf.getvalue()))
    buf.close()
    return img
