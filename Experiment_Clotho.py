import torch
import os
import sys

# custom
from util import *
from transformers import GPT2Tokenizer
from ClipCap_forAAC.CLIPCAP_forAAC import * # network
from Train import *

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
argv_num_with_custom_tokenizer = 2 + 1

if len(sys.argv) == argv_num_with_custom_tokenizer : 
    if sys.argv[2] != 'Custom' :
        print("If you want to train using own vocabulary, input 'Custom' as last argument")
        exit()
elif len(sys.argv) < argv_num_with_gpt2_tokenizer : 
    print("Input experiment name")
    exit()

data_dir = './Clotho'

epochs = 50
LR = 5e-5

# PANNs를 써먹기 위해 prefix_size를 수정
audio_prefix_size = 15
semantic_prefix_size = 11 
prefix_size = audio_prefix_size + semantic_prefix_size

transformer_num_layers = {"temporal_num_layers" : 4, "global_num_layers" : 4}
prefix_size_dict = {"temporal_prefix_size" : audio_prefix_size, "global_prefix_size" : semantic_prefix_size}

vocab_size = None
tokenizer_type = None

if len(sys.argv) == argv_num_with_custom_tokenizer:
    tokenizer = tokenizer_forCustomVocab(Dataset = 'Clotho')
    tokenizer_type = 'Custom'
else :
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_type = 'GPT2'

TEST_BATCH_SIZE = 5
TRAIN_BATCH_SIZE = 75

test_dataloader  = CreateDataloader(tokenizer, data_dir, TEST_BATCH_SIZE, 'evaluation', prefix_size, is_TrainDataset = False, tokenizer_type = tokenizer_type)
train_dataloader = CreateDataloader(tokenizer, data_dir, TRAIN_BATCH_SIZE, 'development', prefix_size, is_TrainDataset = False, tokenizer_type = tokenizer_type)

# control randomness
random_seed = 1000
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#============실험================
torch.cuda.empty_cache()

MODEL_NAME = sys.argv[1] + '_clotho'
if tokenizer_type == 'Custom':
    MODEL_NAME += '_CustomHeader' 

createDirectory(MODEL_NAME)

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

model = get_ClipCap_AAC(tokenizer, 
                        vocab_size = vocab_size, Dataset = 'Clotho',
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = False, decoder_freeze = True,
                        pretrain_fromAudioCaps = True, device = device)

Train(model, LR, train_dataloader, test_dataloader,
    epochs, model_name = MODEL_NAME, beam_search = True, device = device,
    Dataset = 'Clotho')

torch.cuda.empty_cache()
#============실험================