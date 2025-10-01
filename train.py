import os
import time 
import warnings

from models.Model import PM 

from utils.utils import *
from utils.metric import *
from utils.datasets import *
from utils.loss_bank import *

from block.Noise import Noise

warnings.filterwarnings("ignore")

result_folder = c.MODEL_PATH + "/" + time.strftime(c.PROJECT_NAME + "__%H_%M_%S", time.localtime()) + "/"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if not os.path.exists(result_folder): os.mkdir(result_folder)
if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")

#####################
# Model initialize: #
##################### 
pmodel = PM(c.mode) 
pmodel.to(device)

params_trainable = (list(filter(lambda p: p.requires_grad, pmodel.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr)

if c.tain_next:
    model_path = c.MODEL_PATH + "/" + c.CONTINUE_PATH + "/" + "models/" + str(c.CONTINUE_EPOCH) + ".pt"
    load(pmodel, model_path)

setup_logger('train', result_folder, 'logging', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train') 

# Noise Layer
train_noise_layer = Noise(["Combined([SP()])"])
test_noise_layer = Noise(["PCombined([SP()])"])

for epoch in range(c.epochs):
    epoch += 1
        
    loss_history = []
    stego_loss_history = []
    message_loss_history = []
    
    stego_psnr_history = []
    error_history = []

            
    #################
    #     train:    #
    #################
 
    pmodel.train()
    for idx, cover_image in enumerate(trainloader):
        cover_image = cover_image.to(device)
        message = torch.Tensor(np.random.choice([c.min_value, c.max_value], (cover_image.shape[0], c.message_length))).to(device)

        #################
        #    forward:   #
        #################   
        stego_image = pmodel.encoder(cover_image, message)
        stego_loss = stego_loss_fn(stego_image, cover_image, device)

        no_stego_image = train_noise_layer([stego_image.clone(), cover_image.clone()])

        if c.mode == 'DO':
            recover_message = pmodel.decoder(no_stego_image)
            message_loss = do_loss(recover_message, message, c.deflation_weight, c.inflation_weight, c.safe_value, device) 
        elif c.mode == 'PH':
            recover_message, pro_recover_message = pmodel.decoder(no_stego_image)
            message_loss = message_loss_fn(pro_recover_message, message, device)


        stego_loss_history.append(stego_loss.item())  
        stego_psnr = psnr(cover_image, stego_image, 2)
        stego_psnr_history.append(stego_psnr) 
        
        message_loss_history.append(message_loss.item())
        error_rate = decoded_message_error_rate_batch(message, recover_message)
        error_history.append(error_rate) 

      
        total_loss = c.message_weight * message_loss + c.stego_weight * stego_loss
        loss_history.append(total_loss.item())
        
        total_loss.backward()
        optim.step()   
        optim.zero_grad()
         
    
    logger_train.info(
        f"Train epoch: {epoch} | "
        f"Stego Weight: {c.stego_weight} | "
        f'Total Loss: {np.mean(loss_history):.4f} | '
        f'Stego Loss: {np.mean(stego_loss_history):.4f} | ' 
        f'Mess  Loss: {np.mean(message_loss_history):.4f} | '
        f'Stego Psnr: {np.mean(stego_psnr_history):.4f} |'
        f'Acc:  {1 - np.mean(error_history):.4f} |'
    )

    #################
    #     val:      #
    #################
    if epoch % c.val_freq == 0:
        with torch.no_grad():
            test_stego_psnr_history = []
            test_acc = [[]]

            pmodel.eval()
            for test_cover_image in testloader:
                test_cover_image = test_cover_image.to(device)
                test_message = torch.Tensor(np.random.choice([c.min_value, c.max_value], (test_cover_image.shape[0], c.message_length))).to(device)

                #################
                #    forward:   #
                ################# 
                test_stego_image = pmodel.encoder(test_cover_image, test_message)
                test_stego_psnr = psnr(test_cover_image, test_stego_image, 255)
                test_stego_psnr_history.append(test_stego_psnr)
            
                no_test_stego_image_list = test_noise_layer([test_stego_image.clone(), test_cover_image.clone()])
                
                for idx, no_test_stego_image in enumerate(no_test_stego_image_list): 
                    if c.mode == 'DO':
                        test_recover_message = pmodel.decoder(no_test_stego_image)
                    elif c.mode == 'PH':
                        test_recover_message, _ = pmodel.decoder(no_test_stego_image)
                    test_error_rate = decoded_message_error_rate_batch(test_message, test_recover_message)
                    test_acc[idx].append(1 - test_error_rate)
                                    
            
            logger_train.info(
                f"TEST: "
                f'Stego PSNR:  {np.mean(test_stego_psnr_history):.4f} | '
                f'JPEG Acc:  {np.mean(test_acc[0]):.4f} | ')

    print("")
    
    if np.mean(test_acc[0]) > 0.9999:
        c.stego_weight = c.stego_weight * 10

    if (epoch % c.SAVE_freq) == 0:
        if (epoch % c.SAVE_freq) == 0:
            torch.save(pmodel.state_dict(), result_folder + "models/" + str(epoch) + '.pt')


torch.save(pmodel.state_dict(), result_folder + "models/" + str(c.epoch) + '.pt')



    
    
    
    
    
    