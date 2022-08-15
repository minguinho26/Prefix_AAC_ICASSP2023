def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# 여기서부터 프로그램 시작
import sys
import torch
# custom
from AudioCaps.AudioCaps_Dataset_custom_vocab import * # 데이터셋
from ClipCap_forAAC.CLIPCAP_forAAC_custom_vocab import * # network
from Train import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_dir = './AudioCaps'
vocab_size = int(sys.argv[3])
epoch = int(sys.argv[2])
Model_name = sys.argv[1]

tokenizer = tokenizer_AudioCaps(vocab_size)

TEST_BATCH_SIZE = 5

# PANNs를 써먹기 위해 prefix_size를 수정
audio_prefix_size = 15
semantic_prefix_size = 11
prefix_size = audio_prefix_size + semantic_prefix_size

test_dataloader  = dataloader_AudioCapsDataset(data_dir, TEST_BATCH_SIZE, split = 'test', prefix_size = prefix_size, is_TrainDataset = False)

transformer_num_layers = {"audio_num_layers" : 4 , "semantic_num_layers" : 4}
prefix_size_dict = {"audio_prefix_size" : audio_prefix_size, "semantic_prefix_size" : semantic_prefix_size}

model = get_ClipCap_AAC(tokenizer, vocab_size, mapping_type = 'TRANSFORMER', 
prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers)
model.load_state_dict(torch.load("./params_" + Model_name + "/Param_epoch_" + str(epoch) + ".pt"))
eval_model(model, test_dataloader, tokenizer, epoch, Model_name, True)