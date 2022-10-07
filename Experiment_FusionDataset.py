from typing import List, Tuple
import torch
import copy
import os
import sys

# custom
from FusionDataset import * # 데이터셋
from transformers import GPT2Tokenizer
from ClipCap_forAAC.CLIPCAP_forAAC import * # network
from Train import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

# <실험명> <vocabulary의 크기> 를 입력한 경우
if len(sys.argv) == argv_num_with_custom_tokenizer : 
    if (not isNumber(sys.argv[2])) :
        print("<vocabulary의 크기>에 대한 값이 숫자가 아닙니다!")
        exit()
# 따로 입력한 값이 없을 경우
elif len(sys.argv) < argv_num_with_gpt2_tokenizer : 
    print("실험명을 입력해주십시오!")
    exit()

epochs = 50
LR = 5e-5

# PANNs를 써먹기 위해 prefix_size를 수정
audio_prefix_size = 15
semantic_prefix_size = 11 
prefix_size = audio_prefix_size + semantic_prefix_size

transformer_num_layers = {"audio_num_layers" : 4, "semantic_num_layers" : 4}
prefix_size_dict = {"audio_prefix_size" : audio_prefix_size, "semantic_prefix_size" : semantic_prefix_size}

# argv의 개수가 2개다 : custom vocab을 사용했다
vocab_size = None
tokenizer_type = None

if len(sys.argv) == argv_num_with_custom_tokenizer:
    vocab_size = int(sys.argv[2])
    tokenizer = tokenizer_AudioCaps(vocab_size)
    tokenizer_type = 'Custom'
# argv의 개수가 1개다 : custom vocab을 사용하지 않았다
else :
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_type = 'GPT2'

TEST_BATCH_SIZE = 5
TRAIN_BATCH_SIZE = 62
test_dataloader  = dataloader_FusionDataset(tokenizer, TEST_BATCH_SIZE, 'test', prefix_size, is_TrainDataset = False)
train_dataloader = dataloader_FusionDataset(tokenizer, TRAIN_BATCH_SIZE, 'train', prefix_size, is_TrainDataset = True)

#============실험================
torch.cuda.empty_cache()

MODEL_NAME = sys.argv[1] + '_audiocaps'

createDirectory(MODEL_NAME)

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

model = get_ClipCap_AAC(tokenizer, 
                        vocab_size = vocab_size, Dataset = 'AudioCaps',
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = False, decoder_freeze = True,
                        pretrain_fromAudioCaps = False, device = device)

Train(model, LR, train_dataloader, test_dataloader,
    epochs, model_name = MODEL_NAME, beam_search = True, device = device,
    Dataset = 'AudioCaps') # Dataset별 학습전략이 다른데 나는 AudioCaps의 학습전략을 써야겠다.

torch.cuda.empty_cache()
#============실험================