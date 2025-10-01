from torch import nn
from block.combined import Combined, PCombined
from utils.identity import Identity
from utils.mf import MF
from utils.gn import GN
from utils.gf import GF
from utils.sp import SP
from utils.dropout import Dropout
from utils.jpeg import Jpeg, JpegTest, JpegSS, JpegMask
    
class Noise(nn.Module):
    def __init__(self, layers):
        super(Noise, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.noise = nn.Sequential(*layers)

    def forward(self, image_and_cover):
        noised_image = self.noise(image_and_cover)
        return noised_image