import torch.nn as nn

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, images_clean):
        images, clean_image = images_clean
        return images