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
class tokenizer_AudioCaps() :
    
    def encode(self, sentence) :
        
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

        sentence = sentence.strip()
        word_list = sentence.split(' ')
        
        token_idx = []
        for word in word_list : 
            temp_word = word.lower()
            if self.vocab_size == 5217 and temp_word[-1] == ',' :
                temp_word_wo_rest = temp_word[:-1]
                token_idx.append(self.vocab.index(temp_word_wo_rest))
                token_idx.append(self.vocab.index(','))
            else :
                token_idx.append(self.vocab.index(temp_word))

        # 만약 마침표가 반영된 vocab이면 '.' 토큰도 추가
        if self.vocab_size == 6300 :
            token_idx.append(self.vocab.index('.'))
            
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
        
        if vocab_size == 6299 :
            file_path = './AudioCaps/AudioCaps_vocabulary_lower_remove_stopword_len_6299.pickle'
        elif vocab_size == 6300 :
            file_path = './AudioCaps/AudioCaps_vocabulary_lower_remove_stopword_import_stopword_in_vocab_len_6300.pickle'
        elif vocab_size == 5217 :
            file_path = './AudioCaps/AudioCaps_vocabulary_lower_remove_stopword_import_rest_in_vocab_len_5217.pickle'
        
        with open(file_path, 'rb') as f:
            self.vocab = pickle.load(f) 

class AudioCaps_Dataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, prefix_size) :  # split = 'train' or 'test'
        super(AudioCaps_Dataset, self).__init__()
        
        self.SAMPLE_RATE = 16000
        
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
        self.tag_list = []
        self.token_list = []
        
        # audio의 경로, audio에 해당하는 caption을 리스트에 추가
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
            if file[-3:] == 'wav' :
                file_row_in_csv = csv_file[csv_file['youtube_id'] == file[:-4]]
                
                captions = file_row_in_csv['caption'].to_list() # test dataset은 audio 하나에 caption이 5개씩 있음
                for caption in captions : # 1대 1 매칭 되게끔 넣어줌
                    
                    # 문장이 끝난다는걸 나타내는 마침표가 없는 caption이 있다. 
                    # 마침표가 없을 경우 마침표를 따로 넣어주자 
                    if caption[-1] != '.' :
                        caption += '.'
                    
                    # tag에 대한 라벨 생성
                    tag_label = torch.zeros(527).type(torch.FloatTensor)
                    tag_label_idx = audiocaps_tag_dict.get(file[:-4])
                    tag_label[tag_label_idx] = 1
                    
                    self.path_list.append(file)
                    caption_token = tokenizer(caption)['input_ids']
                    self.token_list.append(torch.tensor(caption_token))
                    self.tag_list.append(tag_label)
                    
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
            
        tokens, mask = self.pad_tokens(item)

        # raw audio, gpt2_caption, file_name 출력
        return audio_file, tokens, mask, self.tag_list[item], self.path_list[item]
    

def dataloader_AudioCapsDataset(tokenizer, data_dir, batch_size, split, prefix_size, is_TrainDataset = False) :
    
    dataset = AudioCaps_Dataset(tokenizer, data_dir, split, prefix_size)
    
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







