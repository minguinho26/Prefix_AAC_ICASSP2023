# encdecfreeze_transformer 모델의 성능 측정 

from typing import List, Tuple
import torch
from transformers import GPT2Tokenizer
import copy
import os

# 폴더 생성 메소드
def createDirectory(directory):
    directory = "params_" + directory
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# custom
from AudioCaps_Dataset import * # 데이터셋
from CLIPCAP_forAAC import * # network
from Train import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_dir = './Dataset_16khz'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

TEST_BATCH_SIZE = 25
test_dataloader  = dataloader_AudioCapsDataset(data_dir, TEST_BATCH_SIZE, tokenizer, split = 'test', is_TrainDataset = False )

#============실험 1================
torch.cuda.empty_cache()

TRAIN_BATCH_SIZE = 25 
epochs = 30

train_dataloader = dataloader_AudioCapsDataset(data_dir, TRAIN_BATCH_SIZE, tokenizer, split = 'train', is_TrainDataset = True )

MODEL_NAME = 'clipcap_decfreeze_transformer_only_fbsp_train'
createDirectory(MODEL_NAME)


# 7/28 실험만 'only fbsp trainable'이다. 유의하자
# 실험 돌리고 fbsp만 trainable하게 할지 기존 방식을 쓸지 결정하기
model = get_ClipCap_AAC(tokenizer, mapping_type = 'TRANSFORMER', 
                        encoder_freeze = False, decoder_freeze = True)

min_loss_file_path = Train(model, train_dataloader, test_dataloader, tokenizer,  epochs, model_name = MODEL_NAME, beam_search = True)

torch.cuda.empty_cache()
#============실험 1================


