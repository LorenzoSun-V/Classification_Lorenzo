import random
import cv2
import numpy as np
import torch.nn


class ColorJitter(object):
    def __init__(self, prob=0.5, min_factor=0.5, max_factor=1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob

    def __call__(self, img):
        # undo ColorJitter
        if random.uniform(0, 1) < self.prob:
            return img

        if random.uniform(0, 1) > self.prob:
            img = self.brightness(img)
        if random.uniform(0, 1) > self.prob:
            img = self.saturation(img)
        if random.uniform(0, 1) > self.prob:
            img = self.contrast(img)
        return img

    def brightness(self, img):
        factor = np.random.uniform(self.min_factor, self.max_factor)
        result = img * factor
        if factor > 1:
            result[result > 255] = 255
        result = result.astype(np.uint8)
        return result

    def saturation(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        factor = np.random.uniform(self.min_factor, self.max_factor)

        result = np.zeros(img.shape, dtype=np.float32)
        result[:, :, 0] = img[:, :, 0] * factor + img_gray * (1 - factor)
        result[:, :, 1] = img[:, :, 1] * factor + img_gray * (1 - factor)
        result[:, :, 2] = img[:, :, 2] * factor + img_gray * (1 - factor)
        result[result > 255] = 255
        result[result < 0] = 0
        result = result.astype(np.uint8)
        return result

    def contrast(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_mean = np.mean(img_gray)
        temp = np.ones((img.shape[0], img.shape[1]), dtype=np.float32) * gray_mean
        factor = np.random.uniform(self.min_factor, self.max_factor)

        result = np.zeros(img.shape, dtype=np.float32)
        result[:, :, 0] = img[:, :, 0] * factor + temp * (1 - factor)
        result[:, :, 1] = img[:, :, 1] * factor + temp * (1 - factor)
        result[:, :, 2] = img[:, :, 2] * factor + temp * (1 - factor)

        result[result > 255] = 255
        result[result < 0] = 0
        result = result.astype(np.uint8)
        return result


class ScaleCrop(object):
    """
    With a probability, first increase img size to (1 + 1/8), and then perform random crop.
    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, scale, p=0.5, interpolation=cv2.INTER_CUBIC):
        self.height = height
        self.width = width
        self.scale = scale
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            return cv2.resize(img, (self.width, self.height), self.interpolation)
        new_width, new_height = int(round(self.width * self.scale)), int(round(self.height * self.scale))
        resized_img = cv2.resize(img, (new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img[y1: y1 + self.height, x1: x1 + self.width]
        return croped_img


class Blur(object):
    def __init__(self, prob=0.5, mode='random', kernel_size=3, sigma=1):
        self.prob = prob
        self.mode = mode
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.uniform(0, 1) < self.prob:
            return img

        if self.mode == 'random':
            self.mode = random.choice(['normalized', 'gaussian', 'median'])

        if self.mode == 'normalized':
            result = cv2.blur(img, (self.kernel_size, self.kernel_size))
        elif self.mode == 'gaussian':
            result = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigmaX=self.sigma, sigmaY=self.sigma)
        elif self.mode == 'median':
            result = cv2.medianBlur(img, self.kernel_size)
        else:
            print('Blur mode is not supported: %s.' % self.mode)
            result = img
        return result


class Rotation(object):
    def __init__(self, prob=0.5, degree=10, mode='crop'):
        self.prob = prob
        self.degree = random.randint(1, degree)
        self.mode = mode

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            h, w = img.shape[:2]
            center_x, center_y = w / 2, h / 2
            M = cv2.getRotationMatrix2D((center_x, center_y), self.degree, scale=1)

            if self.mode == 'crop':  # keep original size
                new_w, new_h = w, h
            else:  # keep full img
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int(h * sin + w * cos)
                new_h = int(h * cos + w * sin)
                M[0, 2] += (new_w / 2) - center_x
                M[1, 2] += (new_h / 2) - center_y

            result_img = cv2.warpAffine(img, M, (new_w, new_h))
            return result_img
        else:
            return img


class Flip(object):
    def __init__(self, prob=0.5, mode='h'):
        self.prob = prob
        self.mode = mode

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            if self.mode == 'h':
                return cv2.flip(img, 1)
            elif self.mode == 'v':
                return cv2.flip(img, 0)
            else:
                print('Unsupported mode: %s.' % self.mode)
                return img
        return img


class Resize(object):
    def __init__(self, size_in_pixel=None, size_in_scale=None):
        """
        :param size_in_pixel: tuple (width, height)
        :param size_in_scale: tuple (width_scale, height_scale)
        :return:
        """
        self.size_in_pixel = size_in_pixel
        self.size_in_scale = size_in_scale

    def __call__(self, img):
        if self.size_in_pixel is not None:
            return cv2.resize(img, self.size_in_pixel)
        elif self.size_in_scale is not None:
            return cv2.resize(img, (0, 0), fx=self.size_in_scale[0], fy=self.size_in_scale[1])
        else:
            print('size_in_pixel and size_in_scale are both None.')
            return img
