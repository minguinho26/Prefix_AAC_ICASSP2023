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
from AudioCaps.AudioCaps_Dataset_custom_vocab import *
from ClipCap_forAAC.CLIPCAP_forAAC_custom_vocab import * # network
from Train import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_dir = './Dataset'

TEST_BATCH_SIZE = 5
TRAIN_BATCH_SIZE = 73
epochs = 50
LR = 5e-5

# PANNs를 써먹기 위해 prefix_size를 수정
audio_prefix_size = 15
semantic_prefix_size = 11
prefix_size = audio_prefix_size + semantic_prefix_size

#============실험================
torch.cuda.empty_cache()

# network name 지어주기
architecture_ver = '2_2_final'
experiment_num = 3
case = 2
vocab_size = 6299
tokenizer = tokenizer_AudioCaps(vocab_size)

test_dataloader  = dataloader_AudioCapsDataset(data_dir, TEST_BATCH_SIZE, split = 'test', prefix_size = prefix_size, vocab_size = vocab_size, is_TrainDataset = False)

train_dataloader = dataloader_AudioCapsDataset(data_dir, TRAIN_BATCH_SIZE, split = 'train', prefix_size = prefix_size, vocab_size = vocab_size, is_TrainDataset = True )
MODEL_NAME = 'clipcap_archi_' + architecture_ver + '_experiment_' + str(experiment_num) + '_case_' + str(case)

createDirectory(MODEL_NAME)

transformer_num_layers = {"audio_num_layers" : 4 , "semantic_num_layers" : 4}
prefix_size_dict = {"audio_prefix_size" : audio_prefix_size, "semantic_prefix_size" : semantic_prefix_size}

model = get_ClipCap_AAC(tokenizer, vocab_size, mapping_type = 'TRANSFORMER', prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, encoder_freeze = False, decoder_freeze = True)

min_loss_file_path = Train(model, LR, train_dataloader, test_dataloader, tokenizer, epochs, model_name = MODEL_NAME, beam_search = True)

torch.cuda.empty_cache()