import torch
from astropy.io import fits
import numbers
import numpy as np
import random

class CenterCropFits(object):
    """Crops the given .fits file at the center.

        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.
        """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (fits): Image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        return center_crop_fits(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def center_crop_fits(img, output_size):
    centerx, centery = img.shape[0]/2, img.shape[1]/2
    th, tw = output_size[0], output_size[1]
    croph1, croph2 = int(centery-th/2), int(centery+th/2)
    cropw1, cropw2 = int(centerx-tw/2), int(centerx+tw/2)
    fits_center = img[cropw1:cropw2, croph1:croph2, :]
    return fits_center


class ToTensorFits(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor_fits(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def to_tensor_fits(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img

# cutrange:
class CutRangeFits(object):
    """Condenses the range of a fits to a maximum and a minimum value.
    """

    def __init__(self, mini):
        self.mini = mini

    def __call__(self, pic):
        return cut_range_fits(pic, self.mini)

    def __repr__(self):
        return self.__class__.__name__ + '(mini={0})'.format(self.mini)


def cut_range_fits(pic, mini):
    for row in range(0, len(pic)):
        for element in range(0, len(pic[row])):
            if pic[row][element][0] < mini:
                pic[row][element][0] = mini
            else:
                pass
    return pic


class RandomHorizontalFlipFits(object):
    """Horizontally flip the given Fits randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Fits): Image to be flipped.

        Returns:
            Fits: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.fliplr(img).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipFits(object):
    """Vertically flip the given Fits randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Fits): Image to be flipped.

        Returns:
            Fits: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.flipud(img).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

