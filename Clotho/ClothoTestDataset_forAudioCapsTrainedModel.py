import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
import os
import numpy as np
from tqdm import tqdm
import re

class ClothoTestDataset_forAudioCapsTrainedModel(Dataset):
    def __init__(self, data_dir, prefix_size) :  # split = 'evaluation'
        super(ClothoTestDataset_forAudioCapsTrainedModel, self).__init__()
        
        self.SAMPLE_RATE = 16000
        
        self.change_sampling_rate = torchaudio.transforms.Resample(self.SAMPLE_RATE, 16000)
        
        self.split = 'evaluation'
        
        self.audio_files_dir = data_dir + '/clotho_audio_files/' + self.split
        
        csv_file_path = data_dir + '/clotho_csv_files/' + 'clotho_captions_' + self.split + '.csv'
        
        audio_file_list = os.listdir(self.audio_files_dir)
        
        self.audio_file_list = []
        self.caption_list = []
        
        
        csv_file = pd.read_csv(csv_file_path)
        # audio의 경로, audio에 해당하는 caption을 리스트에 추가
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
            
            
            audio_file_full_path = self.audio_files_dir + '/' + file
            
            audio_file, _ = torchaudio.load(audio_file_full_path)
            audio_file = audio_file.squeeze(0)
            
            set_length = 10
            
            ratio = audio_file.shape[0]/(self.SAMPLE_RATE * set_length)

            compressed_audio_file = torch.zeros(self.SAMPLE_RATE * set_length)

            for idx in range(self.SAMPLE_RATE * set_length) :
                ratio_idx = int(ratio * idx)
                compressed_audio_file[idx] = audio_file[ratio_idx]
            
            self.audio_file_list.append(compressed_audio_file)
            
            caption_list_each_audio = []
            for i in range(5) :
                
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
                
                
                caption_list_each_audio.append(caption)
                    
            self.caption_list.append(caption_list_each_audio)

        self.prefix_length = prefix_size
            
    def __len__(self):
       
        return len(self.audio_file_list)

    
    def __getitem__(self, item: int) :

        return self.audio_file_list[item], self.caption_list[item], self.audio_name_list[item]
    

def dataloader_ClothoTestDataset_forAudioCapsTrainedModel(data_dir, prefix_size) :
    
    dataset = ClothoTestDataset_forAudioCapsTrainedModel(data_dir, prefix_size)

    cpu_core_num = 8
    dataloader = DataLoader(dataset=dataset,
                      batch_size=5,
                      shuffle=False,
                      num_workers=cpu_core_num,
                      drop_last=False)
    
    return dataloader