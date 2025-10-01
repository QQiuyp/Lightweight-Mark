import torch
import numpy as np
import torch.nn as nn
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.image_pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
        ) 

        self.message_T_conv = nn.ConvTranspose2d(1, 6, kernel_size=16, stride=16, padding=0, bias=False)
       
        self.after_concat_layer = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        ) 

    def forward(self, image, message):
        image_pre = self.image_pre_layer(image)
        
        size = int(np.sqrt(message.shape[1]))
        message_image = message.view(-1, 1, size, size)
        
        message_pre = self.message_T_conv(message_image)

        concat1 = torch.cat([image_pre, message_pre], dim=1)

        output = self.after_concat_layer(concat1)

        return output + image