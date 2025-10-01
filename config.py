epochs = 200

# DO related
deflation_weight = 1
inflation_weight = 1
safe_value = -0.1

message_weight = 1
stego_weight = 1

message_length = 64
min_value = -1
max_value = 1   

# Mode:
mode = "DO"

# Train:  
lr = 1e-3
batch_size = 128  
cropsize = 128

# Val:
batchsize_val = 16
cropsize_val = 128
val_freq = 1 
 
# Data Path
TRAIN_PATH = ''
VAL_PATH = ''
 
format_train = 'jpg'
format_val = 'png'  

# Saving checkpoints: 
MODEL_PATH = 'experiments'  
PROJECT_NAME = ""  
SAVE_freq = 2 

# Continue path:
CONTINUE_PATH = 'DO/CombinedNoise_DO'
CONTINUE_EPOCH = 200
tain_next = True
