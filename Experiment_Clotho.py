from typing import List, Tuple
import torch
import copy
import os
import sys
from Clotho.Clotho_Dataset import tokenizer_Clotho

# custom
from Clotho.Clotho_Dataset import * # 데이터셋
from transformers import GPT2Tokenizer
from ClipCap_forAAC.CLIPCAP_forAAC import * # network
from Train import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# 폴더 생성 메소드
def createDirectory(MODEL_NAME):
    directory = "./params_" + MODEL_NAME
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

# <실험명> <vocabulary의 크기> 를 입력한 경우
if len(sys.argv) == argv_num_with_custom_tokenizer : 
    if (not isNumber(sys.argv[2])) :
        print("<vocabulary의 크기>에 대한 값이 숫자가 아닙니다!")
        exit()
# 따로 입력한 값이 없을 경우
elif len(sys.argv) < argv_num_with_gpt2_tokenizer : 
    print("실험명을 입력해주십시오!")
    exit()

data_dir = './Clotho'

epochs = 50
LR = 5e-5

audio_prefix_size = 15
semantic_prefix_size = 11
prefix_size = audio_prefix_size + semantic_prefix_size

transformer_num_layers = {"audio_num_layers" : 4 , "semantic_num_layers" : 4}
prefix_size_dict = {"audio_prefix_size" : audio_prefix_size, "semantic_prefix_size" : semantic_prefix_size}

# argv의 개수가 2 + 1개다 : custom vocab을 사용했다
vocab_size = None
if len(sys.argv) == argv_num_with_custom_tokenizer:
    vocab_size = int(sys.argv[2])
    tokenizer = tokenizer_Clotho(vocab_size)
# argv의 개수가 1개다 : custom vocab을 사용하지 않았다
else :
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

TEST_BATCH_SIZE = 5
TRAIN_BATCH_SIZE = 32 
test_dataloader = dataloader_ClothoDataset(tokenizer, data_dir, TEST_BATCH_SIZE, split = 'evaluation', prefix_size = prefix_size, is_TrainDataset = False)
train_dataloader = dataloader_ClothoDataset(tokenizer, data_dir, TEST_BATCH_SIZE, split = 'development', prefix_size = prefix_size, is_TrainDataset = True)

#============실험================
torch.cuda.empty_cache()

MODEL_NAME = sys.argv[1] + '_clotho'

createDirectory(MODEL_NAME)

model = get_ClipCap_AAC(tokenizer, vocab_size = vocab_size, mapping_type = 'TRANSFORMER', Dataset = 'Clotho',
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = False, decoder_freeze = True,
                        pretrain_fromAudioCaps = True)

Train(model, LR, train_dataloader, test_dataloader, 
    tokenizer, epochs, model_name = MODEL_NAME, beam_search = True,
    Dataset = 'Clotho')

torch.cuda.empty_cache()
