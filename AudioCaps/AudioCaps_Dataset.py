import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
import os
import numpy as np
from tqdm import tqdm
import pickle
import re
import string

# vocabulary만들 때 소문자로 변환만 한 경우 사용되는 tokenizer
class tokenizer_AudioCaps() :
    
    def encode(self, sentence) :
        
        word_list = sentence.split(' ')
        
        token_idx = []
        for word in word_list : 

            if self.vocab_size == 5084 and word[-1] == ',' :
                word_wo_rest = word[:-1]
                token_idx.append(self.vocab.index(word_wo_rest))
                token_idx.append(self.vocab.index(','))
            else :
                token_idx.append(self.vocab.index(word))
            
        # 마지막에 <eos> 추가
        token_idx.append(13)
        
        return token_idx
    
    def decode(self, token_idx) :
        
        sentence = ''
        for idx in token_idx :
            if (idx == 13) :
                break
            else :
                if (self.vocab[idx] == '.') or (self.vocab[idx] == ',') :
                    sentence = sentence[:-1] # 마침표나 쉼표는 공백 제거 후 붙여줘야함
                
                sentence += self.vocab[idx] + ' '
        
        sentence = sentence.rstrip() # 우측 공백 제거

        # 맨 마지막에 마침표 있으면 제거해주기
        if sentence[-1] == '.' :
            sentence = sentence[:-1]
        
        return sentence

    def __init__(self, vocab_size) :
        
        file_path = ''
        self.vocab_size = vocab_size
        
        if vocab_size == 5084 : # 마침표, 쉼표 제거 + 쉼표를 vocab에 포함 (with <unk> token)
            file_path = './AudioCaps/AudioCaps_vocabulary_5084.pickle'
        elif vocab_size == 7911 : # 마침표 제거 X (with <unk> token)
            file_path = './AudioCaps/AudioCaps_vocabulary_7911.pickle'
        elif vocab_size == 5069 : # ACT에서 쓴 문장 처리만 사용 (with <unk> token)
            file_path = './AudioCaps/AudioCaps_vocabulary_5069.pickle'
        elif vocab_size == 4992 : # 모든 기호 제거(=단어만 vocab에 포함)
            file_path = './AudioCaps/AudioCaps_vocabulary_4992.pickle'  
        
        with open(file_path, 'rb') as f:
            self.vocab = pickle.load(f) 

class AudioCaps_Dataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, prefix_size, tokenizer_type = 'GPT2') :  # split = 'train' or 'test'
        super(AudioCaps_Dataset, self).__init__()
        
        self.SAMPLE_RATE = 16000
        self.split = split
        
        # data_dir 은 dataset폴더겠지?
        # dataset폴더 안에 train, test폴더 만들고 각 폴더에 .wav랑 .csv를 넣어야겠다 
        self.data_dir = data_dir + '/' + split + '/' # 'dataset/train/'과 같이 나올거다
        # csv file은 train.csv 혹은 test.csv임
        csv_file = pd.read_csv(self.data_dir + split + '.csv')
        
        # file의 이름이 youtube_id임
        audio_file_list = os.listdir(self.data_dir)
        
        # audio별 tag에 대한 label 휙득을 위함 
        
        tag_file_path = data_dir + '/' + split + '/audiocaps_' + split + '_tag_dict.pickle'
        with open(tag_file_path, 'rb') as f:
            audiocaps_tag_dict = pickle.load(f)
        
        self.path_list = []
        self.token_list = []
        self.caption_list_for_test = []
        
        # audio의 경로, audio에 해당하는 caption을 리스트에 추가
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
            if file[-3:] == 'wav' :
                file_row_in_csv = csv_file[csv_file['youtube_id'] == file[:-4]]
                
                captions = file_row_in_csv['caption'].to_list() # test dataset은 audio 하나에 caption이 5개씩 있음
                for caption in captions : # 1대 1 매칭 되게끔 넣어줌
                    self.path_list.append(file)
                    
                    caption = caption.lower()
                    
                    # 문장 교정================================
                    # 쉼표 오류 제거
                    caption = caption.replace(',', ' , ') 
                    # 공백 줄이기
                    caption = re.sub(' +', ' ', caption)
                    caption = caption.replace(' ,', ',')
                    
                    # 49275개의 caption 중 192개만 뒤에 마침표 있었다 
                    # 마침표는 잚못 넣은 것으로 판단하여 마침표를 제거한다
                    caption = re.sub(r'[.]', '', caption)

                    # 문장 교정만 한 것을 넣는다(즉, 쉼표 등의 기호가 모두 포함)
                    if split != 'train' :
                        self.caption_list_for_test.append(caption)
                    
                    if tokenizer.vocab_size == 5069 :
                        caption = re.sub(r'\s([,.!?;:"](?:\s|$))', r'\1', caption).replace('  ', ' ')
                        caption = re.sub('[,.!?;:\"]', ' ', caption).replace('  ', ' ')
                    elif (tokenizer.vocab_size == 7911) or (tokenizer_type == 'GPT2') or (tokenizer.vocab_size == 5084) :
                        if (tokenizer.vocab_size == 7911) or (tokenizer_type == 'GPT2'):
                            caption += '.'
                    elif tokenizer.vocab_size == 4992 :
                        caption = caption.translate(str.maketrans('', '', string.punctuation))
                    
                    caption = caption.strip()
                    # 문장 교정================================
                    
                    if split == 'train' :
                        if tokenizer_type == 'GPT2' :
                            tokens = tokenizer(caption)['input_ids']
                        else :
                            tokens = tokenizer.encode(caption)

                        self.token_list.append(torch.tensor(tokens))
                        
        if split == 'train' :          
            self.all_len = torch.tensor([len(self.token_list[i]) for i in range(len(self.token_list))]).float()
            self.max_seq_len = min(int(self.all_len.mean() + self.all_len.std() * 10), int(self.all_len.max()))
        self.prefix_length = prefix_size # audio_prefix_length + semantic_prefix_length
            
    def __len__(self):
       
        return len(self.path_list)
    
    def pad_tokens(self, item: int):
        tokens = self.token_list[item].clone().detach()
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.token_list[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.token_list[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
    
    
    def __getitem__(self, item: int) :
        
        audio_file_full_path = self.data_dir + self.path_list[item]
       
        # 모든 audio는 10초로 크기가 통일되어 있음(Sampling Rate : 16kHz)
        # 10초짜리 audio들로 구성됨
        audio_file, _ = torchaudio.load(audio_file_full_path)
        audio_file = audio_file.squeeze(0)
        
        # 지정한 Length를 기준으로 slicing or padding을 수행
        set_length = 10
        
        # slicing
        if audio_file.shape[0] > (self.SAMPLE_RATE * set_length) :
            audio_file = audio_file[:self.SAMPLE_RATE * set_length]
        # zero padding
        if audio_file.shape[0] < (self.SAMPLE_RATE * set_length) :
            pad_len = (self.SAMPLE_RATE * set_length) - audio_file.shape[0]
            pad_val = torch.zeros(pad_len)
            audio_file = torch.cat((audio_file, pad_val), dim=0)
            
        # raw audio, gpt2_caption, file_name 출력
        
        if self.split == 'train' :
            tokens, mask = self.pad_tokens(item)
            return audio_file, tokens, mask, self.path_list[item]
        else :
            return audio_file, self.caption_list_for_test[item], self.path_list[item]
            
            
    

def dataloader_AudioCapsDataset(tokenizer, data_dir, batch_size, split, prefix_size, is_TrainDataset = False, tokenizer_type = 'GPT2') :
    
    dataset = AudioCaps_Dataset(tokenizer, data_dir, split, prefix_size, tokenizer_type = tokenizer_type)
    
    if is_TrainDataset == True :
        is_shuffle = True
        is_drop_last = True
    else :
        is_shuffle = False
        is_drop_last = False
    
    cpu_core_num = 8
    dataloader = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=cpu_core_num,
                      drop_last=is_drop_last)
    
    return dataloader