import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import librosa
import os
import numpy as np
from tqdm import tqdm

class AudioCaps_Dataset(Dataset):
    def __init__(self, data_dir, tokenizer, split) :  # split = 'train' or 'test'
        super(AudioCaps_Dataset, self).__init__()
        
        self.SAMPLE_RATE = 16000
        
        # data_dir 은 dataset폴더겠지?
        # dataset폴더 안에 train, test폴더 만들고 각 폴더에 .wav랑 .csv를 넣어야겠다 
        self.data_dir = data_dir + '/' + split + '/' # 'dataset/train/'과 같이 나올거다
        # csv file은 train.csv 혹은 test.csv임
        csv_file = pd.read_csv(self.data_dir + split + '.csv')
        
        # file의 이름이 youtube_id임
        audio_file_list = os.listdir(self.data_dir)
        
        self.path_list = []
        self.token_list = []
        
        # audio의 경로, audio에 해당하는 caption을 리스트에 추가
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
            if file[-3:] == 'wav' :
                file_row_in_csv = csv_file[csv_file['youtube_id'] == file[:-4]]
                
                captions = file_row_in_csv['caption'].to_list() # test dataset은 audio 하나에 caption이 5개씩 있음

                for caption in captions : # 1대 1 매칭 되게끔 넣어줌
                    self.path_list.append(file)
                    caption_token = tokenizer(caption + '.')['input_ids']
                    self.token_list.append(torch.tensor(caption_token))
                    
        self.all_len = torch.tensor([len(self.token_list[i]) for i in range(len(self.token_list))]).float()
        self.max_seq_len = min(int(self.all_len.mean() + self.all_len.std() * 10), int(self.all_len.max()))
        self.prefix_length = 10
            
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
        audio_file, _ = librosa.load(audio_file_full_path, sr=self.SAMPLE_RATE, dtype=np.float32)
        
        # 지정한 Length를 기준으로 slicing or padding을 수행
        set_length = 10
        
        # slicing
        if audio_file.shape[0] > (self.SAMPLE_RATE * set_length) :
            audio_file = audio_file[:self.SAMPLE_RATE * set_length]
        # zero padding
        if audio_file.shape[0] < (self.SAMPLE_RATE * set_length) :
            pad_len = (self.SAMPLE_RATE * set_length) - audio_file.shape[0]
            audio_file = np.pad(audio_file, (0,pad_len), 'constant', constant_values=0)
        
        tokens, mask = self.pad_tokens(item)

        # raw audio, gpt2_caption, file_name 출력
        return  torch.tensor(audio_file), tokens, mask, self.path_list[item]
    

def dataloader_AudioCapsDataset(data_dir, batch_size, tokenizer, split, is_TrainDataset = False) :
    
    dataset = AudioCaps_Dataset(data_dir, tokenizer, split)
    
    if is_TrainDataset == True :
        is_shuffle = True
        is_drop_last = True
    else :
        is_shuffle = False
        is_drop_last = False
    
    cpu_core_num = 8 # 쿠아이 서버의 cpu는 8코어임
    dataloader = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=cpu_core_num,
                      drop_last=is_drop_last)
    
    return dataloader


