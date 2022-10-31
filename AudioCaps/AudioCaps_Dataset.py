import torch
from torch.utils.data import Dataset

import pandas as pd
import torchaudio
import os
from tqdm import tqdm
import re

import util

class AudioCapsDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, prefix_size, set_length = 10, tokenizer_type = 'GPT2') :  # split = 'train' or 'test'
        super(AudioCapsDataset, self).__init__()
        
        self.SAMPLE_RATE = 16000
        self.split = split
        self.set_length = set_length

        self.data_dir = data_dir + '/' + split + '/'
        csv_file = pd.read_csv(self.data_dir + split + '.csv')
        
        # file's name = youtube_id
        audio_file_list = os.listdir(self.data_dir)
        
        self.path_list = []
        self.token_list = []
        self.caption_list_for_test = []
        
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
            if file[-3:] == 'wav' :
                file_row_in_csv = csv_file[csv_file['youtube_id'] == file[:-4]]
                
                captions = file_row_in_csv['caption'].to_list() # train : 1 caption per each audio, test : 5 captions per each audio
                for caption in captions : 
                    self.path_list.append(file)
                    
                    caption = caption.lower()    
                
                    caption = caption.replace(',', ' , ') 
                    caption = re.sub(' +', ' ', caption)
                    caption = caption.replace(' ,', ',')
                    caption = re.sub(r'[.]', '', caption)

                    caption = caption.strip()

                    caption += '.'
                    
                    if split != 'train' :
                        self.caption_list_for_test.append(caption)
                    elif split == 'train' :
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
       
        audio_file, _ = torchaudio.load(audio_file_full_path)
        audio_file = audio_file.squeeze(0)
        
        # slicing or padding based on set_length
        
        # slicing
        if audio_file.shape[0] > (self.SAMPLE_RATE * self.set_length) :
            audio_file = audio_file[:self.SAMPLE_RATE * self.set_length]
        # zero padding
        if audio_file.shape[0] < (self.SAMPLE_RATE * self.set_length) :
            pad_len = (self.SAMPLE_RATE * self.set_length) - audio_file.shape[0]
            pad_val = torch.zeros(pad_len)
            audio_file = torch.cat((audio_file, pad_val), dim=0)
            
        if self.split == 'train' :
            tokens, mask = self.pad_tokens(item)
            return audio_file, tokens, mask, self.path_list[item]
        else :
            return audio_file, self.caption_list_for_test[item], self.path_list[item]