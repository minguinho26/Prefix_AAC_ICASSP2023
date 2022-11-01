import torch
from torch.utils.data import Dataset

import pandas as pd
import torchaudio
import os
from tqdm import tqdm
import re
import string

import util

class ClothoDataset(Dataset):
    def compress_audio(self, audio, set_length = 10) :
        
        ratio = audio.size()[1]/(self.SAMPLE_RATE * set_length)
        
        compress_idx_list = []
        
        for idx in range(self.SAMPLE_RATE * set_length) :
            compress_idx_list.append(int(ratio * idx))
        
        return audio[:, compress_idx_list]
    
    def __init__(self, tokenizer, data_dir, split, prefix_size, tokenizer_type = 'GPT2', is_settingnum_3 = False) :  # split = 'development' or 'evaluation'
        super(ClothoDataset, self).__init__()
        
        self.SAMPLE_RATE = 16000
        
        self.change_sampling_rate = torchaudio.transforms.Resample(self.SAMPLE_RATE, 16000)
        
        self.split = split
        
        self.audio_files_dir = data_dir + '/clotho_audio_files/' + split
        
        csv_file_path = data_dir + '/clotho_csv_files/' + 'clotho_captions_' + split + '.csv'
        
        audio_file_list = os.listdir(self.audio_files_dir)
        
        self.audio_name_list = []
        self.audio_file_list = []
        self.token_list = []
        self.caption_list_for_test = []
        
        csv_file = pd.read_csv(csv_file_path)
        
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
           
            audio_file_full_path = self.audio_files_dir + '/' + file
            audio_file, _ = torchaudio.load(audio_file_full_path)
            
            if is_settingnum_3 == False :
                audio_file = audio_file.squeeze(0)
                # slicing or padding based on set_length
                set_length = 30

                # slicing
                if audio_file.shape[0] > (self.SAMPLE_RATE * set_length) :
                    audio_file = audio_file[:self.SAMPLE_RATE * set_length]
                # zero padding
                if audio_file.shape[0] < (self.SAMPLE_RATE * set_length) :
                    pad_len = (self.SAMPLE_RATE * set_length) - audio_file.shape[0]
                    pad_val = torch.zeros(pad_len)
                    audio_file = torch.cat((audio_file, pad_val), dim=0)
            else :
                 audio_file = self.compress_audio(audio_file).squeeze(0)
            
            for i in range(5) :
                
                self.audio_file_list.append(audio_file)
                self.audio_name_list.append(file)
                sentence_str = 'caption_' + str(i + 1)
                caption = csv_file[csv_file['file_name'] == file][sentence_str].item()

                caption = caption.lower()    
                
                caption = caption.replace(',', ' , ') 
                caption = re.sub(' +', ' ', caption)
                caption = caption.replace(' ,', ',')
                caption = re.sub(r'[.]', '', caption)
                
                caption = caption.strip()
                caption += '.'
                
                if split != 'development' :
                    self.caption_list_for_test.append(caption)
                elif split == 'development' : 
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
       
        return len(self.audio_file_list)
    
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
        
        if self.split == 'development' : 
            
            tokens, mask = self.pad_tokens(item)
            return self.audio_file_list[item], tokens, mask, self.audio_name_list[item]
        else :
            return self.audio_file_list[item], self.caption_list_for_test[item], self.audio_name_list[item]