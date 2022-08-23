import sys
import torch
# custom
from Clotho.Clotho_Dataset import * # 데이터셋
from transformers import GPT2Tokenizer
from ClipCap_forAAC.CLIPCAP_forAAC import * # network
from Train import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

argv_num_with_gpt2_tokenizer = 2 + 1
argv_num_with_custom_tokenizer = 3 + 1

# <모델명> <몇 번째 epoch에서 학습시켰는지>만 입력한 경우
if len(sys.argv) == argv_num_with_gpt2_tokenizer : 
    if (not isNumber(sys.argv[2])) :
        print("<몇 번째 epoch에서 학습시켰는지>에 대한 값이 숫자가 아닙니다!")
        exit()

  # <모델명> <몇 번째 epoch에서 학습시켰는지> <vocabulary의 크기>를 입력한 경우      
elif len(sys.argv) == argv_num_with_custom_tokenizer :
    if (not isNumber(sys.argv[2])) or (not isNumber(sys.argv[2])) :
        print("<몇 번째 epoch에서 학습시켰는지> 그리고(혹은) <vocabulary의 크기>에 대한 값이 숫자가 아닙니다!")
        exit()
# <모델명> <몇 번째 epoch에서 학습시켰는지>를 제대로 입력하지 않은 경우
elif len(sys.argv) < argv_num_with_gpt2_tokenizer : 
    print("<모델명> <몇 번째 epoch에서 학습시켰는지>를 제대로 입력해주십시오!")
    exit()

data_dir = './Clotho'

epoch = int(sys.argv[2])
Model_name = sys.argv[1]

# PANNs를 써먹기 위해 prefix_size를 수정
audio_prefix_size = 15
# semantic_prefix_size = 11 # 기존의 Semantic mapping network를 사용시
semantic_prefix_size = 10 # 새로운 Semantic mapping network를 사용시
prefix_size = audio_prefix_size + semantic_prefix_size
transformer_num_layers = {"audio_num_layers" : 4 , "semantic_num_layers" : 4}
prefix_size_dict = {"audio_prefix_size" : audio_prefix_size, "semantic_prefix_size" : semantic_prefix_size}

# argv의 개수가 3개다 : custom vocab을 사용했다
if len(sys.argv) == argv_num_with_custom_tokenizer:
    vocab_size = int(sys.argv[3])
    tokenizer = tokenizer_Clotho(vocab_size)
# argv의 개수가 2개다 : custom vocab을 사용하지 않았다
else :
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

TEST_BATCH_SIZE = 5
test_dataloader  = dataloader_ClothoDataset(tokenizer, data_dir, TEST_BATCH_SIZE, split = 'evaluation', prefix_size = prefix_size, is_TrainDataset = False)

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

model = get_ClipCap_AAC(tokenizer, vocab_size = None, Dataset = 'Clotho',
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = False, decoder_freeze = True,
                        pretrain_fromAudioCaps = False, device = device)
model.load_state_dict(torch.load("./Train_record/params_" + Model_name + "_clotho/Param_epoch_" + str(epoch) + ".pt"))
eval_model(model, test_dataloader, tokenizer, epoch, Model_name, True, Dataset = 'Clotho')