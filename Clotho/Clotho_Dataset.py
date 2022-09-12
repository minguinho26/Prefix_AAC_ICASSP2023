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
class tokenizer_Clotho() :
    
    def encode(self, sentence) :
        
        word_list = sentence.split(' ')
        
        token_idx = []
        for word in word_list : 

            if self.vocab_size == 4373 and word[-1] == ',' :
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
        
        return sentence # 앞글자를 대문자로 만들어줌

    def __init__(self, vocab_size) :
        
        file_path = ''
        self.vocab_size = vocab_size
        
        if vocab_size == 4373 : # 마침표, 쉼표 제거 + 쉼표를 vocab에 포함 (with <unk> token)
            file_path = './Clotho/Clotho_vocabulary_4373.pickle'
        elif vocab_size == 7011 : # 마침표 제거 X (with <unk> token)
            file_path = './Clotho/Clotho_vocabulary_7011.pickle'
        elif vocab_size == 4368 : # Clotho == ACT, [walkietalkie walkie-talkie] 말고 구성요소는 다 똑같음 (index별 값도 다 똑같음)
            file_path = './Clotho/Clotho_vocabulary_4368.pickle'
        
        with open(file_path, 'rb') as f:
            self.vocab = pickle.load(f) 

class ClothoDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, prefix_size, tokenizer_type = 'GPT2') :  # split = 'development' or 'evaluation'
        super(ClothoDataset, self).__init__()
        
        self.SAMPLE_RATE = 16000
        
        self.change_sampling_rate = torchaudio.transforms.Resample(self.SAMPLE_RATE, 16000)
        
        self.split = split
        
        self.audio_files_dir = data_dir + '/clotho_audio_files/' + split
        
        csv_file_path = data_dir + '/clotho_csv_files/' + 'clotho_captions_' + split + '.csv'
        
        audio_file_list = os.listdir(self.audio_files_dir)
        
        self.audio_name_list = []
        self.token_list = []
        self.caption_list_for_test = []
        
        csv_file = pd.read_csv(csv_file_path)
        # audio의 경로, audio에 해당하는 caption을 리스트에 추가
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
           
            for i in range(5) :
                
                self.audio_name_list.append(file)
                
                sentence_str = 'caption_' + str(i + 1)
                caption = csv_file[csv_file['file_name'] == file][sentence_str].item()

                caption = caption.lower()
                    
                # 문장 교정================================
                caption = caption.replace(',', ' , ') 
                # 공백 줄이기
                caption = re.sub(' +', ' ', caption)
                caption = caption.replace(' ,', ',')
                
                # Clotho는 가지고 있는 caption에 있는 모든 문장기호를 제거했다
                # 혹시나 마침표가 들어있는 경우 처리해준다
                caption = re.sub(r'[.]', '', caption)
                
                if split != 'development' :
                    self.caption_list_for_test.append(caption)
                        
                if tokenizer.vocab_size == 4368 :
                    caption = caption.translate(str.maketrans('', '', string.punctuation))
                elif (tokenizer.vocab_size == 7011) or (tokenizer_type == 'GPT2') or (tokenizer.vocab_size == 4373) :
                    caption = re.sub(r'[.]', '', caption)
                    if (tokenizer.vocab_size == 7011) or (tokenizer_type == 'GPT2'):
                        caption += '.'
                    
                caption = caption.strip()
                # 문장 교정================================
                
                if split == 'development' :
                    if tokenizer_type == 'GPT2' :
                        tokens = tokenizer(caption)['input_ids']
                    else :
                        tokens = tokenizer.encode(caption)

                    self.token_list.append(torch.tensor(tokens))

                
        if split == 'development' :                           
            self.all_len = torch.tensor([len(self.token_list[i]) for i in range(len(self.token_list))]).float()
            self.max_seq_len = min(int(self.all_len.mean() + self.all_len.std() * 10), int(self.all_len.max()))
        self.prefix_length = prefix_size
            
    def __len__(self):
       
        return len(self.audio_name_list)
    
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
        
        audio_file_full_path = self.audio_files_dir + '/' + self.audio_name_list[item]
        audio_file, _ = torchaudio.load(audio_file_full_path)
        audio_file = audio_file.squeeze(0)
    
        # 지정한 Length를 기준으로 slicing or padding을 수행
        set_length = 30
        
        # slicing
        if audio_file.shape[0] > (self.SAMPLE_RATE * set_length) :
            audio_file = audio_file[:self.SAMPLE_RATE * set_length]
        # zero padding
        if audio_file.shape[0] < (self.SAMPLE_RATE * set_length) :
            pad_len = (self.SAMPLE_RATE * set_length) - audio_file.shape[0]
            pad_val = torch.zeros(pad_len)
            audio_file = torch.cat((audio_file, pad_val), dim=0)
        
        if self.split == 'development' :
            tokens, mask = self.pad_tokens(item)
            return audio_file, tokens, mask, self.audio_name_list[item]
        else :
            return audio_file, self.caption_list_for_test[item], self.audio_name_list[item]
    

def dataloader_ClothoDataset(tokenizer, data_dir, batch_size, split, prefix_size, is_TrainDataset = False, tokenizer_type = 'GPT2') :
    
    dataset = ClothoDataset(tokenizer, data_dir, split, prefix_size, tokenizer_type = tokenizer_type)
    
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