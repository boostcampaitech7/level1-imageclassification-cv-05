#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import timm

class ConvNextModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ConvNextModel, self).__init__()
        self.model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', 
                                       pretrained=pretrained, 
                                       num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

