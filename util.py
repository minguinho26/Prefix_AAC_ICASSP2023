import pickle
import re
import csv
from torch.utils.data import DataLoader

from AudioCaps.AudioCaps_Dataset import *
from Clotho.Clotho_Dataset import *

# Tokenizer of own vocabulary for Training Datset
class tokenizer_forCustomVocab() :
    
    def encode(self, sentence) :
        
        word_list = sentence.split(' ')
        
        token_idx = []
        for word in word_list : 
            token_idx.append(self.vocab.index(word))
            
        #  <eos> 
        token_idx.append(13)
        
        return token_idx
    
    def decode(self, token_idx) :
        
        sentence = ''
        for idx in token_idx :
            if (idx == 13) :
                break
            else :
                sentence += self.vocab[idx] + ' '

        sentence = sentence.rstrip()
        sentence = re.sub(r'[.]', '', sentence)

        sentence += '.'
        
        return sentence

    def __init__(self, Dataset) : # Dataset = 'AudioCaps' or 'Clotho'
        
        file_path = ''

        if Dataset == 'AudioCaps' :
            file_path = './AudioCaps/AudioCaps_vocabulary.pickle'
            with open(file_path, 'rb') as f:
                self.vocab = pickle.load(f) 
        elif Dataset == 'Clotho' :
            file_path = './Clotho/Clotho_vocabulary.pickle'
            with open(file_path, 'rb') as f:
                self.vocab = pickle.load(f) 
        
def CreateDataloader(tokenizer, data_dir, batch_size, split, prefix_size, is_TrainDataset = False, tokenizer_type = 'GPT2') :

    if split == 'train' or split == 'test' :
        dataset = AudioCapsDataset(tokenizer, data_dir, split, prefix_size, set_length = 10, tokenizer_type = tokenizer_type)
    elif split == 'development' or split == 'evaluation' :
        dataset = ClothoDataset(tokenizer, data_dir, split, prefix_size, tokenizer_type = tokenizer_type)

    if is_TrainDataset == True :
        is_shuffle = True
        is_drop_last = True
    else :
        is_shuffle = False
        is_drop_last = False

    cpu_core_num = 8 # num of thread to use for dataloader
    dataloader = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=cpu_core_num,
                      drop_last=is_drop_last)
    
    return dataloader

def get_pred_captions(model, test_dataloader, device, dataset = 'AudioCaps') :
    model.eval()
    
    with open(f"{dataset}_pred_captions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'caption'])
        for i, (audio, captions, f_names) in enumerate(tqdm(test_dataloader, desc="Get Caption...")):
            with torch.no_grad() :
                audio = audio.to(device)
                audio = audio[0,:].unsqueeze(0)

                pred_caption = model(audio, None, beam_search = True)[0][0]

                writer.writerow([f_names[0], pred_caption])