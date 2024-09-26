#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, info_df, transform, is_inference=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # ConvNeXt expects 3-channel input
        image = self.transform(image=image)['image']

        if self.is_inference:
            return image
        else:
            return image, self.targets[index]

