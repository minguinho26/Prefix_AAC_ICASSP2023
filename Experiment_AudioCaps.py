import torch
import os
import sys
import random

# custom
from util import *
from transformers import GPT2Tokenizer
from AAC_Prefix.AAC_Prefix import * # network
from Train import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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


# Folder creation
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

data_dir = './AudioCaps'

epochs = 50
LR = 5e-5

temporal_prefix_size = 15
global_prefix_size = 11 
prefix_size = temporal_prefix_size + global_prefix_size

transformer_num_layers = {"temporal_num_layers" : 4, "global_num_layers" : 4}
prefix_size_dict = {"temporal_prefix_size" : temporal_prefix_size, "global_prefix_size" : global_prefix_size}

vocab_size = None
tokenizer_type = None

if len(sys.argv) == argv_num_with_custom_tokenizer:
    tokenizer = tokenizer_forCustomVocab(Dataset = 'AudioCaps')
    tokenizer_type = 'Custom'
    vocab_size = len(tokenizer.vocab)
else :
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_type = 'GPT2'

# control randomness
random_seed=2766
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
np.random.seed(random_seed)
random.seed(random_seed)  

print("random_seed :", random_seed)
print("vocab_size :", vocab_size)
    
TEST_BATCH_SIZE = 5
TRAIN_BATCH_SIZE = 75

if prefix_size == 0 :
    prefix_size = 26

test_dataloader  = CreateDataloader(tokenizer, data_dir, TEST_BATCH_SIZE, 'test', prefix_size, is_TrainDataset = False, tokenizer_type = tokenizer_type)
train_dataloader = CreateDataloader(tokenizer, data_dir, TRAIN_BATCH_SIZE, 'train', prefix_size, is_TrainDataset = True, tokenizer_type = tokenizer_type)

test_dataloader_clotho = CreateDataloader(tokenizer, './Clotho', TEST_BATCH_SIZE, 'evaluation', prefix_size, is_TrainDataset = False, tokenizer_type = tokenizer_type)


#============Experiment================
torch.cuda.empty_cache()

MODEL_NAME = sys.argv[1] + '_audiocaps'
if tokenizer_type == 'Custom':
    MODEL_NAME += '_CustomHeader' 

createDirectory(MODEL_NAME)

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda' if USE_CUDA else 'cpu')

model = get_AAC_Prefix(tokenizer, 
                        vocab_size = vocab_size, Dataset = 'AudioCaps',
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = False, decoder_freeze = True,
                        pretrain_fromAudioCaps = False, device = device)

Train(model, LR, train_dataloader, test_dataloader,
    epochs, model_name = MODEL_NAME, beam_search = True, device = device,
    Dataset = 'AudioCaps', test_dataloader_other_dataset = test_dataloader_clotho)

torch.cuda.empty_cache()
#============Experiment================