import torch
import sys

# custom
from util import *
from AAC_Prefix.AAC_Prefix import * # network
from Train import *
    
TEST_BATCH_SIZE = 5

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

argv_num = 1 + 3

if len(sys.argv) != argv_num :
    print("you should write 'table_num', 'setting_num' and 'audio file path'!")
    exit()

table_num = sys.argv[1]
setting_num = sys.argv[2]
audio_file_path = sys.argv[3]

# table_num = 1 : Evaluation on Clotho
# table_num = 2 : Evaluation on AudioCaps

# setting_num = 1 : train dataset == test dataset
# setting_num = 2 : train dataset != test dataset
# setting_num = 3 : overall datasets(Clotho & AudioCaps) <- need to test by using compressed audio

if setting_num == 3 :
    is_settingnum_3 = True
else : 
    is_settingnum_3 = False

model = get_model_in_table(table_num, setting_num, device)


# prepare audio input=========
SAMPLE_RATE = 16000
set_length = 30

audio_file, _ = torchaudio.load(audio_file_path)

# slicing
if audio_file.shape[0] > (SAMPLE_RATE * set_length) :
    audio_file = audio_file[:SAMPLE_RATE * set_length]
# zero padding
if audio_file.shape[0] < (SAMPLE_RATE * set_length) :
    pad_len = (SAMPLE_RATE * set_length) - audio_file.shape[0]
    pad_val = torch.zeros(pad_len)
    audio_file = torch.cat((audio_file, pad_val), dim=0)
# prepare audio input=========

if len(audio_file.size()) == 3 :
    audio_file = audio_file.unsqueeze(0)

pred_caption = model(audio_file, None, beam_search = True)[0][0]

print("Caption :", pred_caption)