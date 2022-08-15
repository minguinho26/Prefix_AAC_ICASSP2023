import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import pandas as pd
import torchaudio
import os
import numpy as np
from tqdm import tqdm
import pickle
import re

class ClothoDataset(Dataset):
    def __init__(self, data_dir, split, prefix_size) :  # split = 'development' or 'evaluation'
        super(ClothoDataset, self).__init__()
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
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
                
                sentence_str = 'caption_' + str(i + 1)
                sentence = csv_file[csv_file['file_name'] == file][sentence_str].item()

                # 마침표 제거
                sentence = re.sub(r'[.]', '', sentence)

                # 쉼표 오류 제거
                sentence = sentence.replace(',', ' , ') 

                # 공백 줄이기
                sentence = re.sub(' +', ' ', sentence)

                sentence = sentence.replace(' ,', ',')

                # caption의 마지막이 쉼표일 경우 제거
                if sentence[-1] == ',' :
                    sentence = sentence[:-1]
                    
                # 문장이 끝난다는걸 나타내는 마침표가 없는 caption이 있다. 
                # 마침표가 없을 경우 마침표를 따로 넣어주자 
                if sentence[-1] != '.' :
                    sentence += '.'

                sentence = sentence.strip()
                
                tokens = tokenizer.encode(sentence)
                
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
    

def dataloader_ClothoDataset(data_dir, batch_size, split, prefix_size, is_TrainDataset = False) :
    
    dataset = ClothoDataset(data_dir, split, prefix_size)
    
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