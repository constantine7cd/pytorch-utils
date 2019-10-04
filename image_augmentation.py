from imgaug import augmenters as iaa
import numpy as np


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)