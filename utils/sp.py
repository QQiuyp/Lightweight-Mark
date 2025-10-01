import torch
import torch.nn as nn
import numpy as np
from utils.utils import clamp

class SP(nn.Module):
    def __init__(self, amount=0.1, salt_vs_pepper=0.5):
        super(SP, self).__init__()
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def sp_noise(self, image, amount, salt_vs_pepper):
        device = image.device
        p = amount
        q = salt_vs_pepper
        
        one_mask = torch.zeros_like(image)
        zero_mask = torch.zeros_like(image)
        
        
        for i in range(image.shape[0]):
            flipped = np.random.choice([True, False], size=image.shape[2:], p=[p, 1 - p])
            salted = np.random.choice([True, False], size=image.shape[2:], p=[q, 1 - q])
            peppered = ~salted
            
            one_mask[i,:] = torch.from_numpy((flipped & salted).astype(int))
            zero_mask[i,:] = torch.from_numpy((flipped & peppered).astype(int))
        
        one_mask.to(device)
        zero_mask.to(device)
        
        ones = torch.ones_like(image).to(device)
        zeros = torch.zeros_like(image).to(device)
        image = torch.where(one_mask == 1, ones, image)
        image = torch.where(zero_mask == 1, zeros, image)

        return image

    def forward(self, images_clean):
        image, clean_image = images_clean
        image = (image / 2) + 0.5
        image = self.sp_noise(image, self.amount, self.salt_vs_pepper)
        image = (image - 0.5) * 2
        return image
