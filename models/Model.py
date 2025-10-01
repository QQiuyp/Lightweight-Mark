import torch.nn as nn
from block.Encoder import Encoder
from block.Decoder import Decoder
from block.Noise import Noise


class PM(nn.Module):
    def __init__(self, mode):
        super(PM, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(mode)

    def forward(self, image, message):
        stego_image = self.encoder(image, message)
        recover_message = self.decoder(stego_image)
        return stego_image, recover_message