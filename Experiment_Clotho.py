from typing import List, Tuple
import torch
import copy
import os

# 폴더 생성 메소드
def createDirectory(MODEL_NAME):
    directory = "./params_" + MODEL_NAME
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# custom
from Clotho.Clotho_Dataset import * # 데이터셋
from transformers import GPT2Tokenizer
from ClipCap_forAAC.CLIPCAP_forAAC import * # network
from Train import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_dir = './Dataset'

# 220812 다시 회귀 시도
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

TEST_BATCH_SIZE = 5

audio_prefix_size = 15
semantic_prefix_size = 11
prefix_size = audio_prefix_size + semantic_prefix_size

test_dataloader = dataloader_ClothoDataset(data_dir, TEST_BATCH_SIZE, split = 'evaluation', prefix_size = prefix_size, is_TrainDataset = False)


#============실험================
torch.cuda.empty_cache()

# network name 지어주기
architecture_ver = '2_Final'
experiment_num = 4
case = 3

TRAIN_BATCH_SIZE = 32 
epochs = 8
LR = 5e-5
train_dataloader = dataloader_ClothoDataset(data_dir, TRAIN_BATCH_SIZE, split = 'development', prefix_size = prefix_size, is_TrainDataset = True )

# clotho로 학습시키는거니까 뒤에 'clotho'를 추가해줌 
MODEL_NAME = 'clipcap_archi_' + architecture_ver + '_experiment_' + str(experiment_num) + '_case_' + str(case) + '_clotho'

createDirectory(MODEL_NAME)

transformer_num_layers = {"audio_num_layers" : 4 , "semantic_num_layers" : 4}
prefix_size_dict = {"audio_prefix_size" : audio_prefix_size, "semantic_prefix_size" : semantic_prefix_size}

model = get_ClipCap_AAC(tokenizer, mapping_type = 'TRANSFORMER', prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, encoder_freeze = False, decoder_freeze = True, pretrain_fromAudioCaps = True, PreTrained_Header = True)

model.load_state_dict(torch.load("./params_" + MODEL_NAME + "/Param_epoch_41.pt"))

min_loss_file_path = Train(model, LR, train_dataloader, test_dataloader, tokenizer, epochs, model_name = MODEL_NAME, beam_search = True)

torch.cuda.empty_cache()
#============실험================