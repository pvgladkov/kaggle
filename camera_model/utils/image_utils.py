import StringIO

import numpy as np
from PIL import Image, ImageEnhance
from skimage import exposure
from colour_demosaicing import (mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_bilinear,
                                demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007)


def transform_im(im, crop_type=None, alter_type=None, rotate_angle=None, shape=224):
    alter_types = {
        'resize05': lambda x: resize(x, 0.5),
        'resize08': lambda x: resize(x, 0.8),
        'resize15': lambda x: resize(x, 1.5),
        'resize20': lambda x: resize(x, 2.0),
        'gamma08': lambda x: gamma2(x, 0.8),
        'gamma12': lambda x: gamma2(x, 1.2),
        'q70': lambda x: jpeg_comp(x, 70),
        'q90': lambda x: jpeg_comp(x, 90)
    }

    if rotate_angle > 0:
        im = rotate(im, rotate_angle)

    im = alter_types.get(alter_type, lambda x: x)(im)

    if crop_type is not None:
        im = find_best_crop(im, shape)

    w, h = im.size
    assert w == shape, '{} != {}'.format(w, shape)
    assert h == shape, '{} != {}'.format(h, shape)

    return im


def find_best_crop(im, shape):
    best_crop = None
    best_q = 0
    for i in range(10):
        crop_image = im.crop(random_crop(im.size[0], im.size[1], shape))
        q = crop_quality(crop_image)
        if q > 0.65:
            best_crop = crop_image
            break
        elif q > best_q:
            best_crop = crop_image
            best_q = q
    return best_crop


def random_crop(w, h, size):
    x_margin = min(round(np.random.rand() * w), w - size)
    y_margin = min(round(np.random.rand() * h), h - size)
    return x_margin, y_margin, x_margin + size, y_margin + size


def crop_quality(im):
    img = np.array(im)
    img = img / 255.0
    a = 0.7
    b = 4
    y = np.log(0.01)
    q = 0
    for i in range(3):
        m = np.mean(img[:, :, i])
        std = np.std(img[:, :, i])
        q += a*b*(m - np.power(m, 2)) + (1 - a)*(1 - np.power(np.e, y*std))
    return q / 3.


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


def read_prediction_crop(f_path, shape):
    im = Image.open(f_path)
    im = crop(im, 0, 512)
    im_crop = find_best_crop(im, shape)
    im_array = np.array(im_crop, dtype="uint8")
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
    return Image.fromarray(exposure.adjust_gamma(im, gamma=g))


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


def demosaicing_error(im, mtype='bilinear'):

    types = {
        'bilinear': demosaicing_CFA_Bayer_bilinear,
        'malvar': demosaicing_CFA_Bayer_Malvar2004,
        'menon': demosaicing_CFA_Bayer_Menon2007
    }

    mosaic_im = mosaicing_CFA_Bayer(im)
    demosaic_im = types[mtype](mosaic_im)
    return im - demosaic_im
