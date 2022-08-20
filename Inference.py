import sys

from Clotho.Clotho_Dataset import * # 데이터셋
from transformers import GPT2Tokenizer
from ClipCap_forAAC.CLIPCAP_forAAC import * # network

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

argv_num = 3 + 1

# <학습시킨 Dataset> <vocabulary 종류> <audio file의 경로>를 받음

if len(sys.argv) != argv_num : 
    print("<학습시킨 Dataset> <vocabulary 종류> <audio file의 경로>를 순서대로 입력해주세요!")
    exit()

dataset_type = sys.argv[1]
vocab_type = sys.argv[2]
audio_file_path = sys.argv[3]

if not ((dataset_type == 'AudioCaps') or (dataset_type == 'Clotho')) :
    print("<학습시킨 Dataset>으로 AudioCaps나 Clotho을 입력해주세요!")
    exit()

if not ((vocab_type != 'GPT2') or (vocab_type != 'Custom')) :
    print("<vocabulary 종류>로 GPT2나 Custom을 입력해주세요!")
    exit()

# PANNs를 써먹기 위해 prefix_size를 수정
audio_prefix_size = 15
semantic_prefix_size = 11
prefix_size = audio_prefix_size + semantic_prefix_size
transformer_num_layers = {"audio_num_layers" : 4 , "semantic_num_layers" : 4}
prefix_size_dict = {"audio_prefix_size" : audio_prefix_size, "semantic_prefix_size" : semantic_prefix_size}

# vocab 뭐 쓰는지 판단
vocab_size = None
if vocab_type == 'GPT2' :
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
elif vocab_type == 'Custom' :
    if dataset_type == 'AudioCaps' :
        vocab_size = '알아서 정하기'
    elif dataset_type == 'Clotho' :
        vocab_size = '알아서 정하기'
    tokenizer = tokenizer_Clotho(vocab_size)

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:0' if USE_CUDA else 'cpu') 

model = get_ClipCap_AAC(tokenizer, vocab_size = vocab_size, mapping_type = 'TRANSFORMER', Dataset = dataset_type,
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = False, decoder_freeze = True,
                        pretrain_fromAudioCaps = False)

# 학습시킨 model의 .pt 경로 설정
# model_Clotho_GPT2.pt, model_Clotho_custom.pt
# model_AudioCaps_GPT2.pt, model_AudioCaps_custom.pt
# 4개 중에 하나임
trained_params_path = './Trained_model_params/model_' + dataset_type + '_' + vocab_type + '.pt'
model.load_state_dict(torch.load(trained_params_path))

# inference를 위한 오디오 추출

SAMPLE_RATE = 16000
audio_file, sr = torchaudio.load(audio_file_path)

if sr != SAMPLE_RATE :
    change_sampling_rate = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    audio_file = change_sampling_rate(audio_file).squeeze(0)
else :
    audio_file = audio_file.squeeze(0)

# 지정한 Length를 기준으로 slicing or padding을 수행
set_length = 10

# slicing()
# 개인적으로 slicing보다는 compress가 더 좋은거 같다. 압축된 형태로라도 전체 오디오를 사용할 수 있는거니까. 근데 compress를 구현하는 코드가 안보인다. 
if audio_file.shape[0] > (SAMPLE_RATE * set_length) :
    audio_file = audio_file[:SAMPLE_RATE * set_length]
# zero padding
if audio_file.shape[0] < (SAMPLE_RATE * set_length) :
    pad_len = (SAMPLE_RATE * set_length) - audio_file.shape[0]
    pad_val = torch.zeros(pad_len)
    audio_file = torch.cat((audio_file, pad_val), dim=0)

# caption 생성
model.eval()

audio = audio_file.unsqueeze(0).to(device)

pred_caption = model(audio, None, beam_search = True)[0][0]

print("========Result========")
print("File_path :", audio_file_path)
print("Pred_caption :", pred_caption)
print("========Result========")
