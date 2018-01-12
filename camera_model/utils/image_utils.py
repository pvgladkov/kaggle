import StringIO

import numpy as np
from PIL import Image, ImageEnhance
from skimage import exposure


def transform_im(im, crop_type=None, alter_type=None, rotate_angle=None, shape=224):
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

    im = crop(im, 0, 2*shape)

    if rotate_angle > 0:
        im = rotate(im, rotate_angle)

    im = alter_types.get(alter_type, lambda x: x)(im)

    if crop_type is not None:
        im = crop(im, crop_type, shape)

    return im


def read_and_resize(f_path):
    im_array = np.array(Image.open(f_path), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    return new_array / 255.0


def read_and_crop(f_path, crop_type=None, shape=None):
    im = Image.open(f_path)

    if crop_type is not None and shape is not None:
        im = crop(im, crop_type, shape)

    im_array = np.array(im, dtype="uint8")
    return im_array / 255.0


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


def crop(im, t, size):
    w, h = im.size
    h_size = size // 2
    crop_types = [
        (w // 2 - h_size, h // 2 - h_size, w // 2 + h_size, h // 2 + h_size),
        (0, 0, size, size),
        (w - size, 0, w, size),
        (0, h - size, size, h),
        (w - size, h - size, w, h),
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
