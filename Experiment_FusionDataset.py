import torch
import os
import sys
import random

# custom
from FusionDataset import * # 데이터셋
from transformers import GPT2Tokenizer
from AAC_Prefix.AAC_Prefix import * # network
from Train import *

# reproducibility
def initialization(seed = 0):   
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 

# 폴더 생성 메소드
def createDirectory(MODEL_NAME):
    directory = "./Train_record/params_" + MODEL_NAME
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

argv_num_with_gpt2_tokenizer = 1 + 1


if len(sys.argv) != argv_num_with_gpt2_tokenizer : 
    print("Input experiment name as argument")
    exit()

epochs = 50
LR = 5e-5

temporal_prefix_size = 15
global_prefix_size = 11 
prefix_size = temporal_prefix_size + global_prefix_size

transformer_num_layers = {"temporal_num_layers" : 4, "global_num_layers" : 4}
prefix_size_dict = {"temporal_prefix_size" : temporal_prefix_size, "global_prefix_size" : global_prefix_size}

# argv의 개수가 2개다 : custom vocab을 사용했다
vocab_size = None
tokenizer_type = None

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer_type = 'GPT2'

TEST_BATCH_SIZE = 5
TRAIN_BATCH_SIZE = 62
test_dataloader  = dataloader_FusionDataset(tokenizer, TEST_BATCH_SIZE, 'test', prefix_size, is_TrainDataset = False)
train_dataloader = dataloader_FusionDataset(tokenizer, TRAIN_BATCH_SIZE, 'train', prefix_size, is_TrainDataset = True)

# control randomness
number = 2766
print("random seed : ", 2766)

initialization(seed = 2766)  

#============실험================
torch.cuda.empty_cache()

MODEL_NAME = sys.argv[1] + '_audiocaps'

createDirectory(MODEL_NAME)

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

model = get_AAC_Prefix(tokenizer, 
                        vocab_size = vocab_size, Dataset = 'AudioCaps',
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = False, decoder_freeze = True,
                        pretrain_fromAudioCaps = False, device = device)

Train(model, LR, train_dataloader, test_dataloader,
    epochs, model_name = MODEL_NAME, beam_search = True, device = device,
    Dataset = 'Fusion')

torch.cuda.empty_cache()
#============실험================