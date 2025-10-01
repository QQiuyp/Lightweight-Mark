import torch
import numpy as np
import torch.nn as nn

class Mod(nn.Module): 
    def __init__(self):
        super(Mod, self).__init__() 
        self.mod = nn.Sequential(  
            nn.ConvTranspose2d(1, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True), 
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True), 
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True), 
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),    
 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=2, stride=2, padding=0, bias=False),     
        )

    def forward(self, image):          
        return self.mod(image)
     
class Project(nn.Module): 
    def __init__(self):
        super(Project, self).__init__() 
        self.ADD1 = Mod()
        self.ADD2 = Mod()
        self.ADD3 = Mod()
        self.ADD4 = Mod()
        
        self.MUT1 = Mod() 
        self.MUT2 = Mod()
        self.MUT3 = Mod()
        self.MUT4 = Mod()
      
    def forward(self, image):  
        image1 = image * torch.exp(self.MUT1(image)) + self.ADD1(image)
        image2 = image1 * torch.exp(self.MUT2(image1)) + self.ADD2(image1)
        image3 = image2 * torch.exp(self.MUT3(image2)) + self.ADD3(image2)
        image4 = image3 * torch.exp(self.MUT4(image3)) + self.ADD4(image3)

        return image4 
    
class Decoder(nn.Module): 
    def __init__(self, mode):
        super(Decoder, self).__init__() 
        self.mode = mode
        self.message_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True), 
            
            nn.Conv2d(in_channels=12, out_channels=1, kernel_size=16, stride=16, padding=0, bias=False),   
        )
        
        if self.mode == 'PH':
            self.pro = Project()

        
    def forward(self, image):      
        
        batch_size = image.shape[0]        
        recover_message = self.message_conv(image)    
        
        re_recover_message = recover_message.reshape(batch_size, -1)    
        if self.mode == 'PH':
            pro_recover_message = self.pro(recover_message)
            re_pro_recover_message = pro_recover_message.reshape(batch_size, -1)  
            return re_pro_recover_message 
        elif self.mode == 'DO':
            return re_recover_message  
                 
