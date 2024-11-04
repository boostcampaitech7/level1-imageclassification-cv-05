#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
from PIL import Image, ImageOps
import albumentations as A

class SketchAutoAugment(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(SketchAutoAugment, self).__init__(always_apply, p)
        self.policy = self.sketch_policy()

    def sketch_policy(self):
        return [
            [('Rotate', 0.7, 2), ('Posterize', 0.6, 3)],
            [('ShearX', 0.8, 4), ('AutoContrast', 0.4, None)],
            [('TranslateX', 0.8, 8), ('TranslateY', 0.6, 6)],
            [('Invert', 0.3, None), ('Equalize', 0.5, None)]
        ]

    def apply_augment(self, image, op_name, magnitude):
        img = Image.fromarray(image)
        if op_name == 'Rotate':
            return np.array(img.rotate(magnitude * 2.0))
        elif op_name == 'Posterize':
            return np.array(ImageOps.posterize(img, magnitude))
        elif op_name == 'ShearX':
            return np.array(img.transform(img.size, Image.AFFINE, (1, magnitude * 0.1, 0, 0, 1, 0)))
        elif op_name == 'AutoContrast':
            return np.array(ImageOps.autocontrast(img))
        elif op_name == 'TranslateX':
            return np.array(img.transform(img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0)))
        elif op_name == 'TranslateY':
            return np.array(img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude)))
        elif op_name == 'Invert':
            return np.array(ImageOps.invert(img))
        elif op_name == 'Equalize':
            return np.array(ImageOps.equalize(img))
        return image

    def apply(self, image, **params):
        sub_policy = random.choice(self.policy)
        for op_name, prob, magnitude in sub_policy:
            if random.random() < prob:
                image = self.apply_augment(image, op_name, magnitude)
        return image

    def get_transform_init_args_names(self):
        return ("always_apply", "p")

