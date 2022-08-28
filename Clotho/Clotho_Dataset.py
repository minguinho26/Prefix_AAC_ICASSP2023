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

# vocabulary만들 때 소문자로 변환만 한 경우 사용되는 tokenizer
class tokenizer_Clotho() :
    
    def encode(self, sentence) :
        
        if self.vocab_size == 7556 :
            sentence += '.'
        
        word_list = sentence.split(' ')
        
        token_idx = []
        for word in word_list : 
            temp_word = word.lower()
            if self.vocab_size == 4372 and temp_word[-1] == ',' :
                temp_word_wo_rest = temp_word[:-1]
                token_idx.append(self.vocab.index(temp_word_wo_rest))
                token_idx.append(self.vocab.index(','))
            else :
                token_idx.append(self.vocab.index(temp_word))
            
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
        
        return sentence.capitalize() # 앞글자를 대문자로 만들어줌

    def __init__(self, vocab_size) :
        
        file_path = ''
        self.vocab_size = vocab_size
        
        if vocab_size == 4372 :
            file_path = './Clotho/Clotho_vocabulary_4372.pickle'
        elif vocab_size == 7556 :
            file_path = './Clotho/Clotho_vocabulary_7556.pickle'
        
        with open(file_path, 'rb') as f:
            self.vocab = pickle.load(f) 

class ClothoDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, prefix_size, tokenizer_type = 'GPT2') :  # split = 'development' or 'evaluation'
        super(ClothoDataset, self).__init__()
        
        self.SAMPLE_RATE = 44100
        
        self.change_sampling_rate = torchaudio.transforms.Resample(self.SAMPLE_RATE, 16000)
        
        
        self.audio_files_dir = data_dir + '/clotho_audio_files/' + split
        
        csv_file_path = data_dir + '/clotho_csv_files/' + 'clotho_captions_' + split + '.csv'
        
        audio_file_list = os.listdir(self.audio_files_dir)
        
        self.audio_name_list = []
        self.token_list = []
        all_audio_token_list = []
        
        csv_file = pd.read_csv(csv_file_path)
        # audio의 경로, audio에 해당하는 caption을 리스트에 추가
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
            
            self.audio_name_list.append(file)
            
            token_list_inEachAudio = []
            for i in range(5) :
                
                caption_str = 'caption_' + str(i + 1)
                caption = csv_file[csv_file['file_name'] == file][caption_str].item()

                # 문장 교정================================
                # 마침표 제거
                caption = re.sub(r'[.]', '', caption) 

                # 쉼표 오류 제거
                caption = caption.replace(',', ' , ') 

                # 공백 줄이기
                caption = re.sub(' +', ' ', caption)

                caption = caption.replace(' ,', ',')
                    
                # caption의 마지막이 쉼표일 경우 제거
                if caption[-1] == ',' :
                    caption = caption[:-1]

                caption = caption.strip()
                # 문장 교정================================
                
                if tokenizer_type == 'GPT2' :
                    caption += '.'
                    tokens = tokenizer(caption)['input_ids']
                else :
                    tokens = tokenizer.encode(caption)
                
                token_list_inEachAudio.append(tokens)
                all_audio_token_list.append(tokens)
            
            self.token_list.append(token_list_inEachAudio)     
                                   
        self.all_len = torch.tensor([len(all_audio_token_list[i]) for i in range(len(all_audio_token_list))]).float()
        self.max_seq_len = min(int(self.all_len.mean() + self.all_len.std() * 10), int(self.all_len.max()))
        self.prefix_length = prefix_size
            
    def __len__(self):
       
        return len(self.audio_name_list)
    
    def pad_tokens(self, item: int):
        
        temp_mask_list = None
        temp_token_list = None
        
        for token in self.token_list[item] :
            temp_token = torch.tensor(token)
            padding = self.max_seq_len - temp_token.shape[0]
            if padding > 0:
                temp_token = torch.cat((temp_token, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                temp_token = temp_token[:self.max_seq_len]
                
            mask = temp_token.ge(0)  # mask is zero where we out of sequence
            temp_token[~mask] = 0
            mask = mask.float()
            mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
            
            if temp_mask_list == None :
                temp_mask_list = mask
                temp_token_list = temp_token
            else : 
                temp_mask_list = torch.cat((temp_mask_list, mask), dim = 0)
                temp_token_list = torch.cat((temp_token_list, temp_token), dim = 0)
        
        temp_mask_list = temp_mask_list.view(5, -1)
        temp_token_list = temp_token_list.view(5, -1)
            
        return temp_token_list, temp_mask_list
    
    
    def __getitem__(self, item: int) :
        
        audio_file_full_path = self.audio_files_dir + '/' + self.audio_name_list[item]
       
        audio_file, _ = torchaudio.load(audio_file_full_path)
        audio_file = self.change_sampling_rate(audio_file).squeeze(0)
    
        # 지정한 Length를 기준으로 slicing or padding을 수행
        set_length = 30
        
        change_sample_rate = 16000
        
        # slicing
        if audio_file.shape[0] > (change_sample_rate * set_length) :
            audio_file = audio_file[:change_sample_rate * set_length]
        # zero padding
        if audio_file.shape[0] < (change_sample_rate * set_length) :
            pad_len = (change_sample_rate * set_length) - audio_file.shape[0]
            pad_val = torch.zeros(pad_len)
            audio_file = torch.cat((audio_file, pad_val), dim=0)
            
        tokens, mask = self.pad_tokens(item)
        
        # raw audio, gpt2_caption, file_name 출력
        return audio_file, tokens.type(torch.int64), mask.type(torch.int64), self.audio_name_list[item]
    

def dataloader_ClothoDataset(tokenizer, data_dir, batch_size, split, prefix_size, is_TrainDataset = False) :
    
    dataset = ClothoDataset(tokenizer, data_dir, split, prefix_size)
    
    if is_TrainDataset == True :
        is_shuffle = True
        is_drop_last = True
    else :
        is_shuffle = False
        is_drop_last = False
        
    cpu_core_num = 6
    dataloader = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=cpu_core_num,
                      drop_last=is_drop_last)
    
    return dataloader