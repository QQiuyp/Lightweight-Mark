import os
import warnings
import torchvision

from models.Model import PM 

from utils.utils import *
from utils.metric import *
from utils.datasets import *
from utils.loss_bank import *

from block.Noise import Noise


warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################
# Model initialize: #
#####################


pmodel = PM('DO')
pmodel.cuda()
pmodel.eval()

model_path = c.MODEL_PATH + "/" + c.CONTINUE_PATH + "/" + "models/" + str(c.CONTINUE_EPOCH) + ".pt"
load(pmodel, model_path)


noise_layer = Noise(["PCombined([JpegTest(50), MF(7), GF(2.0), GN(0.05), SP(0.1), Dropout(0.4), Identity()])"])


#################
#     test:     #
################# 
with torch.no_grad():
    stego_psnr_history = []  
    acc = [[], [], [], [], [], [], []]
    for i in range(40):
        for idx, test_cover_image in enumerate(testloader):
            test_cover_image = test_cover_image.to(device) 
            test_message = torch.Tensor(np.random.choice([c.min_value, c.max_value], (test_cover_image.shape[0], c.message_length))).to(device)
        
            ################# 
            #    forward:   # 
            #################
            test_stego_image = pmodel.encoder(test_cover_image, test_message)
            test_stego_image = (test_stego_image - test_cover_image) * 1.0 + test_cover_image
            psnr_temp_stego = psnr(test_cover_image, test_stego_image, 255)
            stego_psnr_history.append(psnr_temp_stego)
        
            no_test_stego_image_list = noise_layer([test_stego_image.clone(), test_cover_image.clone()])
            for idx, no_test_stego_image in enumerate(no_test_stego_image_list):
                test_recover_message = pmodel.decoder(no_test_stego_image)
                error_rate = decoded_message_error_rate_batch(test_message, test_recover_message, 0, 0)
                acc[idx].append((1 - error_rate)*100)        

        
            for i in range(test_cover_image.shape[0]):
                number = 1 + i + idx * test_cover_image.shape[0] 
                torchvision.utils.save_image((((test_stego_image[i] - test_cover_image[i])*10 / 2)+0.5),
                                             os.path.join("Show","{}.png".format(number)))


                
    print(f"TEST:   "
          f'PSNR_STEGO: {np.mean(stego_psnr_history):.2f} | '
          f'Jpeg Acc: {np.mean(acc[0]):.2f} | '
          f'M  F Acc: {np.mean(acc[1]):.2f} | '
          f'G  F Acc: {np.mean(acc[2]):.2f} | '
          f'G  N Acc: {np.mean(acc[3]):.2f} | '
          f'S  P Acc: {np.mean(acc[4]):.2f} | '
          f'D  P Acc: {np.mean(acc[5]):.2f} | '
          f'I  D Acc: {np.mean(acc[6]):.2f} | '
          f'AVEG Acc: {(np.mean(acc[0])+np.mean(acc[1])+np.mean(acc[2])+np.mean(acc[3])+np.mean(acc[4])+np.mean(acc[5]))/6:.4f} | ')

